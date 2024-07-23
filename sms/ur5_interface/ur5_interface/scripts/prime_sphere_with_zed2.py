import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from ur5py.ur5 import UR5Robot
from raftstereo.zed_stereo import Zed
from autolab_core import RigidTransform, DepthImage, CameraIntrinsics, PointCloud, RgbCloud
from convert_poses_to_json import save_poses
import open3d as o3d
from tqdm import tqdm
import viser
import pdb
import pyvista as pv
import argparse
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pyzed.sl as sl

HOME_DIR = "/home/lifelong/sms/sms/ur5_interface/ur5_interface"
# wrist_to_cam = RigidTransform.load("/home/lifelong/ur5_legs/T_webcam_wrist.tf")
wrist_to_cam = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
world_to_extrinsic_zed = RigidTransform.load('/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf')
# threshold to filte
nerf_frame_to_image_frame = np.array([[1,0,0,0],
                                        [0,-1,0,0],
                                        [0,0,-1,0],
                                        [0,0,0,1]])

def visualize_poses(poses, radius=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for pose in poses:
        translation = pose.translation
        rotation = pose.rotation
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        z_axis = rotation[:, 2]

        # Plot the X, Y, Z axes of the frame
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="r",
            length=0.01,
            normalize=True,
        )
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="g",
            length=0.01,
            normalize=True,
        )
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="b",
            length=0.01,
            normalize=True,
        )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)


def get_hemi_translations(
    phi_min, phi_max, theta_min, theta_max, table_center, phi_div, theta_div, R
):
    sin, cos = lambda x: np.sin(np.deg2rad(x)), lambda x: np.cos(np.deg2rad(x))
    rel_pos = np.zeros((phi_div * theta_div, 3))
    for i, phi in enumerate(np.linspace(phi_min, phi_max, phi_div)):
        tmp_pose = []
        for j, theta in enumerate(np.linspace(theta_min, theta_max, theta_div)):
            tmp_pose.append(
                np.array(
                    [R * sin(phi) * cos(theta), R * sin(phi) * sin(theta), R * cos(phi)]
                )
            )
        if i % 2 == 1:
            tmp_pose.reverse()
        for k, pose in enumerate(tmp_pose):
            rel_pos[i * theta_div + k] = pose

    return rel_pos + table_center


def get_rotation(point, center):
    direction = point - center
    z_axis = direction / np.linalg.norm(direction)

    x_axis_dir = -np.cross(np.array((0, 0, 1)), z_axis)
    if np.linalg.norm(x_axis_dir) < 1e-10:
        x_axis_dir = np.array((0, 1, 0))
    x_axis = x_axis_dir / np.linalg.norm(x_axis_dir)
    y_axis_dir = np.cross(z_axis, x_axis)
    y_axis = y_axis_dir / np.linalg.norm(y_axis_dir)

    R = RigidTransform.rotation_from_axes(x_axis, y_axis, z_axis)
    return R


def point_at(cam_t, obstacle_t, extra_R=np.eye(3)):
    """
    cam_t: numpy array of 3D position of gripper
    obstacle_t: numpy array of 3D position of location to point camera at
    """
    direction = obstacle_t - cam_t
    z_axis = direction / np.linalg.norm(direction)
    x_axis_dir = -np.cross(np.array((0, 0, 1)), z_axis)
    if np.linalg.norm(x_axis_dir) < 1e-10:
        x_axis_dir = np.array((0, 1, 0))
    x_axis = x_axis_dir / np.linalg.norm(x_axis_dir)
    y_axis_dir = np.cross(z_axis, x_axis)
    y_axis = y_axis_dir / np.linalg.norm(y_axis_dir)

    # postmultiply the extra rotation to rotate the camera WRT itself
    R = RigidTransform.rotation_from_axes(x_axis, y_axis, z_axis)
    return R


def save_pose(base_to_wrist, i=0, save_dir=None):
    base_to_wrist.from_frame = "wrist"
    base_to_wrist.to_frame = "base"

    # stupid Kush convention
    wrist_to_cam.from_frame = "cam"
    wrist_to_cam.to_frame = "wrist"
    # wrist_to_cam_flipped = cam_to_wrist
    # wrist_to_cam_flipped.from_frame ='cam_flipped'

    # cam_flipped_to_cam = RigidTransform(np.array([[-1,0,0],[0,-1,0],[0,0,1]]),np.zeros(3),from_frame='cam',to_frame='cam_flipped')

    cam_to_nerfcam = RigidTransform(
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        np.zeros(3),
        from_frame="nerf_cam",
        to_frame="cam",
    )

    # cam_pose = (base_to_wrist * wrist_to_cam_flipped * cam_flipped_to_cam) * cam_to_nerfcam
    cam_pose = base_to_wrist * wrist_to_cam * cam_to_nerfcam

    if save_dir is not None:
        np.savetxt(os.path.join(save_dir, f"{i:03d}.txt"), cam_pose.matrix)

    return cam_pose.matrix


def set_up_dirs(scene_name):
    img_save_dir = f"{HOME_DIR}/data/{scene_name}/img"
    img_r_save_dir = f"{HOME_DIR}/data/{scene_name}/img_r"
    depth_save_dir = f"{HOME_DIR}/data/{scene_name}/depth"
    # no_t_depth_save_dir = f"{HOME_DIR}/data/{scene_name}/no_table_depth"
    # no_t_depth_viz_save_dir = f"{HOME_DIR}/data/{scene_name}/no_table_depth_png"
    # seg_mask_save_dir = f"{HOME_DIR}/data/{scene_name}/seg_mask"
    depth_viz_save_dir = f"{HOME_DIR}/data/{scene_name}/depth_png"
    pose_save_dir = f"{HOME_DIR}/data/{scene_name}/poses"

    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(img_r_save_dir, exist_ok=True)
    os.makedirs(depth_save_dir, exist_ok=True)
    # os.makedirs(no_t_depth_save_dir, exist_ok=True)
    # os.makedirs(no_t_depth_viz_save_dir, exist_ok=True)
    # os.makedirs(seg_mask_save_dir, exist_ok=True)
    os.makedirs(depth_viz_save_dir, exist_ok=True)
    os.makedirs(pose_save_dir, exist_ok=True)
    return {
        "img": img_save_dir,
        "img_r": img_r_save_dir,
        "depth": depth_save_dir,
        # "no_table_depth": no_t_depth_save_dir,
        # "no_table_depth_png": no_t_depth_viz_save_dir,
        "depth_png": depth_viz_save_dir,
        # "seg_mask": seg_mask_save_dir,
        "poses": pose_save_dir,
    }


def save_imgs(img_l, img_r, depth, i, save_dirs, flip_table=False):
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_dirs["img"], f"{i:03d}.png"), img_l)
    cv2.imwrite(os.path.join(save_dirs["img_r"], f"{i:03d}.png"), img_r)
    np.save(os.path.join(save_dirs["depth"], f"{i:03d}.npy"), depth)
    plt.imsave(os.path.join(save_dirs["depth_png"], f"{i:03d}.png"), depth,cmap='jet')
    # if flip_table:
    #     np.save(os.path.join(save_dirs["no_table_depth"], f"{i:03d}.npy"), no_t_depth)
    #     plt.imsave(
    #         os.path.join(save_dirs["no_table_depth_jpg"], f"{i:03d}.png"), no_t_depth
    #     )
    # np.save(os.path.join(save_dirs["seg_mask"], f"{i:03d}.npy"), seg_mask)


def table_rejection_depth(depth, camera_intr, transform):
    depth_im = DepthImage(depth, frame="zed")
    point_cloud = camera_intr.deproject(depth_im)
    point_cloud = PointCloud(point_cloud.data, frame="zed")
    tsfm = RigidTransform(
        *RigidTransform.rotation_and_translation_from_matrix(transform),
        from_frame="zed",
        to_frame="base",
    )
    point_cloud = tsfm * point_cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.data.T)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    point = np.delete(point_cloud.data.T, inliers, axis=0)
    # z_average = np.min(point_cloud.data.T[inliers, 2])
    point = np.delete(point, np.where(point[:, 2] < -0.05), axis=0)

    pc = tsfm.inverse() * PointCloud(point[:, :3].T, frame="base")
    depth_im = camera_intr.project_to_image(pc).raw_data[:, :, 0]

    mask = np.zeros_like(depth_im)
    neg_depth = np.where(depth_im <= 0)
    depth_im[neg_depth] = 0.6
    mask[neg_depth] = 1

    hi_depth = np.where(depth_im > 0.8)
    depth_im[hi_depth] = 0.6
    mask[hi_depth] = 1

    cloud = DepthImage(depth_im, "zed").point_normal_cloud(camera_intr).point_cloud
    plotter = pv.Plotter()
    plotter.add_points(cloud.data.T, color="blue")
    # plotter.add_points(removed.data.T, color="red")
    # plotter.show()
    return depth_im, np.array(mask, dtype=bool)

def isolateTable(cam,proper_world_to_cam):
    img_l, img_r = cam.get_rgb()
    depth,points,rgbs  = cam.get_depth_image_and_pointcloud(img_l,img_r,from_frame="cam")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.data.T)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    table_points = points.data.T[inliers]
    db = DBSCAN(eps=0.005, min_samples=10).fit(table_points)
    label_set = set(db.labels_)
    label_set.remove(-1)
    print(label_set)
    max_area = 0
    x_min_cam = 0
    x_max_cam = 0
    y_min_cam = 0
    y_max_cam = 0
    z_min_cam = 0
    z_max_cam = 0
    for label in label_set:
        filtered_table_point_mask = db.labels_ == label#(db.labels_ != -1)
        filtered_table_pointcloud = points.data.T[inliers][filtered_table_point_mask]
        min_bounding_cube_camera_frame = np.array([np.min(filtered_table_pointcloud[:,0]),np.min(filtered_table_pointcloud[:,1]),np.min(points.data.T[:,2]),1]).reshape(-1,1)
        max_bounding_cube_camera_frame = np.array([np.max(filtered_table_pointcloud[:,0]),np.max(filtered_table_pointcloud[:,1]),np.max(filtered_table_pointcloud[:,2]),1]).reshape(-1,1)
        x_min_cam_cluster = min_bounding_cube_camera_frame[0,0]
        y_min_cam_cluster = min_bounding_cube_camera_frame[1,0]
        z_min_cam_cluster = min_bounding_cube_camera_frame[2,0]
        x_max_cam_cluster = max_bounding_cube_camera_frame[0,0]
        y_max_cam_cluster = max_bounding_cube_camera_frame[1,0]
        z_max_cam_cluster = max_bounding_cube_camera_frame[2,0]
        hull = ConvexHull(filtered_table_pointcloud)
        if(hull.volume > max_area):
            max_area = hull.volume
            x_min_cam = x_min_cam_cluster
            x_max_cam = x_max_cam_cluster
            y_min_cam = y_min_cam_cluster
            y_max_cam = y_max_cam_cluster
            z_min_cam = z_min_cam_cluster
            z_max_cam = z_max_cam_cluster

    global_pointcloud = proper_world_to_cam.apply(points)
    
    min_bounding_cube_camera_frame = np.array([x_min_cam,y_min_cam,z_min_cam,1]).reshape(-1,1)
    max_bounding_cube_camera_frame = np.array([x_max_cam,y_max_cam,z_max_cam,1]).reshape(-1,1)
    min_bounding_cube_world = proper_world_to_cam.matrix @ min_bounding_cube_camera_frame
    max_bounding_cube_world = proper_world_to_cam.matrix @ max_bounding_cube_camera_frame
    #offset = np.array([0.01,0.01,0.01,0]).reshape(-1,1)
    min_bounding_cube_world = min_bounding_cube_world# - offset
    max_bounding_cube_world = max_bounding_cube_world# + offset
    
    x_min_world = min_bounding_cube_world[0,0]
    y_min_world = min_bounding_cube_world[1,0]
    z_min_world = min_bounding_cube_world[2,0]
    x_max_world = max_bounding_cube_world[0,0]
    y_max_world = max_bounding_cube_world[1,0]
    z_max_world = max_bounding_cube_world[2,0]
    if(x_min_world > x_max_world):
        temp = x_min_world
        x_min_world = x_max_world
        x_max_world = temp
    if(y_min_world > y_max_world):
        temp = y_min_world
        y_min_world = y_max_world
        y_max_world = temp
    if(z_min_world > z_max_world):
        temp = z_min_world
        z_min_world = z_max_world
        z_max_world = temp
    return x_min_world,x_max_world,y_min_world,y_max_world,z_min_world,z_max_world
    
def convert_pointcloud_to_image(points,rgbs,K,image_width,image_height):
    ones = np.ones((points.data.T.shape[0],1))
    homogenous_points = np.hstack((points.data.T,ones))
    projected_points = K @ homogenous_points.T
    projected_pixels = projected_points[:2] / projected_points[2]
    projected_pixels = projected_pixels.T
    projected_pixels = np.round(projected_pixels).astype('int')
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image_mask = np.zeros((image_height,image_width),dtype=np.uint8)
    depth_image = np.zeros((image_height,image_width),dtype='float32')
    for i in range(projected_pixels.shape[0]):
        x, y = int(projected_pixels[i, 0]), int(projected_pixels[i, 1])
        if 0 <= x < image_width and 0 <= y < image_height:
            image[y, x] = rgbs.data.T[i]
            image_mask[y,x] = 255
            depth_image[y,x] = points.data.T[i,2]
    
    image_mask_inverted = cv2.bitwise_not(image_mask)
    image_inpainted = cv2.inpaint(image,image_mask_inverted,3,cv2.INPAINT_TELEA)
    depth_inpainted = cv2.inpaint(depth_image,image_mask_inverted,3,cv2.INPAINT_TELEA)
    
    cv2.imwrite('/home/lifelong/image2.png',image_mask_inverted)
    cv2.imwrite('/home/lifelong/image1.png',image)
    cv2.imwrite('/home/lifelong/image3.png',image_inpainted)
    return image_inpainted,depth_inpainted

def prime_sphere_main(scene_name, single_image=False, flip_table=False):
    debug = False
    save_dirs = set_up_dirs(scene_name)
    use_robot, use_cam = True, True
    if use_robot:
        
        robot = UR5Robot(gripper=1)
        clear_tcp(robot)
        home_joints = np.array([0.3103832006454468, -1.636097256337301, -0.5523660818683069, -2.4390090147601526, 1.524283766746521, 0.29816189408302307])
        robot.move_joint(home_joints,vel=1.0,acc=0.1)
        world_to_wrist = robot.get_pose()
        world_to_wrist.from_frame = "wrist"
        world_to_cam = world_to_wrist * wrist_to_cam
        proper_world_to_cam_translation = world_to_cam.translation
        proper_world_to_cam_rotation = np.array([[0,1,0],[1,0,0],[0,0,-1]])
        proper_world_to_cam = RigidTransform(rotation=proper_world_to_cam_rotation,translation=proper_world_to_cam_translation,from_frame='cam',to_frame='world')
        proper_world_to_wrist = proper_world_to_cam * wrist_to_cam.inverse()
        
        robot.move_pose(proper_world_to_wrist,vel=1.0,acc=0.1)
        #home_joints = np.array([0.007606238126754761, -1.5527289549456995, -1.573484245930807, -1.599767033253805, 1.610335350036621, 0.020352210849523544])
        # # safe if moving from suction home pose
        #robot.move_joint(home_joints)
        robot.gripper.open()
    if use_cam:    
        zed1 = Zed()
        zed2 = Zed()
        zed_mini_focal_length = 730 # For 1280x720
        cam = None
        extrinsic_zed = None
        save_joints = False
        saved_joints = []
        
        if(abs(zed1.f_ - zed_mini_focal_length) < abs(zed2.f_ - zed_mini_focal_length)):
            cam = zed1
            extrinsic_zed = zed2
        else:
            cam = zed2
            extrinsic_zed = zed1
        # cam.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 10)
        # cam.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 60)
        # print("Zed mini Exposure is set to: ",
        #     cam.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
        # )
        # print("Zed mini Gain is set to: ",
        #     cam.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN),
        # )
        # extrinsic_zed.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 10)
        # extrinsic_zed.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 65)
        # print("Extrinsic Zed Exposure is set to: ",
        #     extrinsic_zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
        # )
        # print("Extrinsic Zed Gain is set to: ",
        #     extrinsic_zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN),
        # )
    global_pointcloud = None
    global_rgbcloud = None
    img_l,img_r = extrinsic_zed.get_rgb()
    depth,points,rgbs  = extrinsic_zed.get_depth_image_and_pointcloud(img_l,img_r,from_frame="zed_extrinsic")
    K = np.array([[cam.f_,0,cam.cx_,0],[0,cam.f_,cam.cy_,0],[0,0,1,0]])
    image_width,image_height = cam.width_,cam.height_
    
    image_inpainted,depth_inpainted = convert_pointcloud_to_image(points,rgbs,K,image_width,image_height)
   
    points_world_frame = world_to_extrinsic_zed.apply(points)
    global_pointcloud = points_world_frame.data.T
    global_rgbcloud = rgbs.data.T
    if debug:
        debug_server = viser.ViserServer()
        debug_server.add_point_cloud('extrinsic_pc',points=global_pointcloud,colors=global_rgbcloud,point_size=0.001)
        img_l,img_r = cam.get_rgb()
        wrist_pose = robot.get_pose()
        wrist_pose.from_frame = 'wrist'
        # print(pose)
        print(wrist_pose)
        cam_pose = wrist_pose * wrist_to_cam
        depth,points,rgbs  = cam.get_depth_image_and_pointcloud(img_l,img_r,from_frame="cam")
            
        points_world_frame = cam_pose.apply(points)
        debug_server.add_point_cloud('top_down_pc',points=points_world_frame.data.T,colors=rgbs.data.T,point_size=0.001)
    
    world_to_extrinsic_zed_image_frame = world_to_extrinsic_zed.matrix @ nerf_frame_to_image_frame
    world_to_extrinsic_zed_image_rigid_tf = RigidTransform(rotation=world_to_extrinsic_zed_image_frame[:3,:3],translation=world_to_extrinsic_zed_image_frame[:3,3],from_frame="zed_extrinsic",to_frame="world")
    np.savetxt(os.path.join(save_dirs["poses"], "000.txt"), world_to_extrinsic_zed_image_rigid_tf.matrix)
    save_imgs(
                    image_inpainted,
                    image_inpainted,
                    depth_inpainted,
                    0,
                    save_dirs,
                    flip_table=flip_table,
                ) 

    # pdb.set_trace()
    tool_to_wrist = RigidTransform()
    # need to set to zero so the frame is at the wrist joint
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
    x_min_world,x_max_world,y_min_world,y_max_world,z_min_world,z_max_world = isolateTable(cam,proper_world_to_cam)

    # angle range from top of sphere
    phi_min, phi_max = 65, 20
    theta_min, theta_max = 125, -125
    phi_div, theta_div = 4, 12
    table_center = np.array([0.48666, -0.0104, -0.120])
    radius = 0.35

    translations = get_hemi_translations(
        phi_min, phi_max, theta_min, theta_max, table_center, phi_div, theta_div, radius
    )
    rotations = [point_at(translation, table_center) for translation in translations]
    poses = [
        RigidTransform(rotations[i], translations[i], from_frame="cam")
        * wrist_to_cam.inverse()
        for i in range(len(translations))
    ]
    start_pose = robot.get_pose()
    start_pose.from_frame='cam'
    poses.insert(0,start_pose)
    visualize_poses(poses)

    # radius -= 0.05
    # translations = get_hemi_translations(
    #     phi_min, phi_max, theta_min, theta_max, table_center, phi_div, theta_div, radius
    # )
    # rotations = [point_at(translation, table_center) for translation in translations]
    # poses.extend(
    #     [
    #         RigidTransform(rotations[i], translations[i], from_frame="cam")
    #         * cam_to_wrist.inverse()
    #         for i in range(len(translations))
    #     ]
    # )
    left_images = []
    right_images = []
    world_to_images = []
    if use_robot:
        for i, pose in enumerate(tqdm(poses)):
            print("Before:", pose)
            robot.move_pose(pose, vel=0.4)
            
            time.sleep(1.0)

            wrist_pose = robot.get_pose()
            # print(pose)
            print(wrist_pose)
            cam_pose = save_pose(wrist_pose, i+1, save_dirs["poses"])
            world_to_image_frame = cam_pose @ nerf_frame_to_image_frame
            print("From frame: " + str("image_frame_"+str(i+1)))
            world_to_image_rigid_tf = RigidTransform(rotation=world_to_image_frame[:3,:3],translation=world_to_image_frame[:3,3],from_frame="image_frame_"+str(i+1),to_frame="world")
            world_to_images.append(world_to_image_rigid_tf)
            if use_cam:
                img_l, img_r = cam.get_rgb()
                left_images.append(img_l)
                right_images.append(img_r)
                
                # cam_intr = cam.get_intr()
                # no_t_depth, seg_mask = table_rejection_depth(
                #     depth,
                #     cam_intr,
                #     cam_pose,
                # )
                # save_imgs(
                #     img_l,
                #     img_r,
                #     depth,
                #     # no_t_depth,
                #     # seg_mask,
                #     i,
                #     save_dirs,
                #     flip_table=flip_table,
                # ) 
            time.sleep(0.05)
    
    # First image comes from Zed 2
    # i = 1
    
    if use_cam:
        camera_intr = cam.get_intr()
        camera_intr.save(f"{HOME_DIR}/data/{scene_name}/zed.intr")
        save_poses(save_dirs["poses"], cam.get_ns_intrinsics(), wrist_to_cam)
    i = 1
    for (left_image,right_image,world_to_image) in zip(left_images,right_images,world_to_images):
        depth,points,rgbs  = cam.get_depth_image_and_pointcloud(left_image,right_image,from_frame="image_frame_"+str(i))
        pointcloud_world_frame = world_to_image.apply(points)
        if(global_pointcloud is None):
            global_pointcloud = pointcloud_world_frame.data.T
            global_rgbcloud = rgbs.data.T
        else:
            if(debug):
                debug_server.add_point_cloud('hemisphere_photo_'+str(i-1),pointcloud_world_frame.data.T,rgbs.data.T,visible=False,point_size=0.001)
                import pdb
                pdb.set_trace()
            global_pointcloud = np.vstack((global_pointcloud,pointcloud_world_frame.data.T))
            global_rgbcloud = np.vstack((global_rgbcloud,rgbs.data.T))
            
        save_imgs(
                    left_image,
                    right_image,
                    depth,
                    i,
                    save_dirs,
                    flip_table=flip_table,
                ) 
        print("Made depth image " + str(i) + "/" + str(len(left_images)))
        i += 1
    
    close_pointcloud = global_pointcloud[(global_pointcloud[:, 0] >= x_min_world) & (global_pointcloud[:, 0] <= x_max_world) &
        (global_pointcloud[:, 1] >= y_min_world) & (global_pointcloud[:, 1] <= y_max_world) &
        (global_pointcloud[:, 2] >= z_min_world) & (global_pointcloud[:, 2] <= z_max_world)
    ]
    close_rgbcloud = global_rgbcloud[(global_pointcloud[:, 0] >= x_min_world) & (global_pointcloud[:, 0] <= x_max_world) &
        (global_pointcloud[:, 1] >= y_min_world) & (global_pointcloud[:, 1] <= y_max_world) &
        (global_pointcloud[:, 2] >= z_min_world) & (global_pointcloud[:, 2] <= z_max_world)
    ]
    num_gaussians_initialization = 200000
    close_gaussian_indices = np.random.choice(close_pointcloud.shape[0],num_gaussians_initialization,replace=False)
    subsampled_close_pointcloud = close_pointcloud[close_gaussian_indices]
    subsampled_close_rgbcloud = close_rgbcloud[close_gaussian_indices]
    db = DBSCAN(eps=0.005, min_samples=20) #
    labels = db.fit_predict(subsampled_close_pointcloud)
    subsampled_close_pointcloud = subsampled_close_pointcloud[labels != -1]
    subsampled_close_rgbcloud = subsampled_close_rgbcloud[labels != -1]
    not_close_pointcloud = global_pointcloud[~((global_pointcloud[:, 0] >= x_min_world) & (global_pointcloud[:, 0] <= x_max_world) &(global_pointcloud[:, 1] >= y_min_world) & (global_pointcloud[:, 1] <= y_max_world) &(global_pointcloud[:, 2] >= z_min_world) & (global_pointcloud[:, 2] <= z_max_world))]
    not_close_rgbcloud = global_rgbcloud[~((global_pointcloud[:, 0] >= x_min_world) & (global_pointcloud[:, 0] <= x_max_world) &(global_pointcloud[:, 1] >= y_min_world) & (global_pointcloud[:, 1] <= y_max_world) &(global_pointcloud[:, 2] >= z_min_world) & (global_pointcloud[:, 2] <= z_max_world))]
    not_close_gaussian_initialization = num_gaussians_initialization * not_close_pointcloud.shape[0] / close_pointcloud.shape[0]
    not_close_gaussian_indices = np.random.choice(not_close_pointcloud.shape[0],int(not_close_gaussian_initialization),replace=False)
    subsampled_not_close_pointcloud = not_close_pointcloud[not_close_gaussian_indices]
    subsampled_not_close_rgbcloud = not_close_rgbcloud[not_close_gaussian_indices]
    full_subsampled_pointcloud = np.vstack((subsampled_close_pointcloud,subsampled_not_close_pointcloud))
    full_subsampled_rgbcloud = np.vstack((subsampled_close_rgbcloud,subsampled_not_close_rgbcloud))
    final_indices = np.random.choice(full_subsampled_pointcloud.shape[0],num_gaussians_initialization,replace=False)
    final_pointcloud = full_subsampled_pointcloud[final_indices]
    final_rgbcloud = full_subsampled_rgbcloud[final_indices]
    server = viser.ViserServer()
    server.add_point_cloud(name="full_pointcloud",points=final_pointcloud,colors=final_rgbcloud,point_size=0.001)
    # db = DBSCAN(eps=0.005, min_samples=20) #
    # labels = db.fit_predict(subsampled_pointcloud)
    # subsampled_pointcloud = subsampled_pointcloud[labels != -1]
    # subsampled_rgbcloud = subsampled_rgbcloud[labels != -1]
    # server.add_point_cloud(name="pointcloud_dbscan",points=subsampled_pointcloud,colors=subsampled_rgbcloud,point_size=0.001)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(final_rgbcloud / 255.)
    o3d.io.write_point_cloud(os.path.join(save_dirs['poses'],'..','sparse_pc.ply'),pcd)
    table_bounding_cube = {
        'x_min':x_min_world,
        'x_max':x_max_world,
        'y_min':y_min_world,
        'y_max':y_max_world,
        'z_min':z_min_world,
        'z_max':z_max_world
    }
    import pdb
    pdb.set_trace()
    with open(os.path.join(save_dirs['poses'],'..','table_bounding_cube.json'), 'w') as json_file:
        json.dump(table_bounding_cube, json_file, indent=4)
    # collection_finish_joints = np.array(robot.get_joints())
    # collection_finish_joints[-1] = -np.pi / 2
    # collection_finish_joints[-2] = np.pi / 2
    # robot.move_joint(collection_finish_joints)

    # # Moving into suction pose
    # collection_finish_joints[-3] = np.pi
    # robot.move_joint(collection_finish_joints)
    # collection_finish_joints[-2] = -np.pi / 2
    # collection_finish_joints[-1] = 0
    # robot.move_joint(collection_finish_joints)

    # robot.kill()

    # collection_finish_joints = np.array(robot.get_joints())
    # collection_finish_joints[-1] = -np.pi / 2
    # collection_finish_joints[-2] = np.pi / 2
    # robot.move_joint(collection_finish_joints)

    # # Moving into suction pose
    # collection_finish_joints[-3] = np.pi
    # robot.move_joint(collection_finish_joints)
    # collection_finish_joints[-2] = -np.pi / 2
    # collection_finish_joints[-1] = 0
    # robot.move_joint(collection_finish_joints)

    robot.kill()
    input("Kill Pointcloud?")
    return 1


import json


def table_paste_dir(scene_name):
    with open(os.path.join(scene_name, "transforms.json")) as r:
        transform_json = json.load(r)
        print(transform_json)
    cam_intrinsics = CameraIntrinsics.load("zed.intr")
    for i, frame_dict in enumerate(tqdm(transform_json["frames"])):
        path = os.path.join(
            scene_name, "depth", frame_dict["file_path"].split("/")[-1][:-3] + "npy"
        )
        depth = np.load(path)
        # NOTE: I changed stuff below this comment -Karim 
        # new_depth, mask = table_rejection_depth(
        #     depth, cam_intrinsics, np.array(frame_dict["transform_matrix"])
        # )
        plt.imshow(depth)
        plt.savefig(f"{HOME_DIR}/data/{scene_name}/depth_png/{i:03d}.png")
        # plt.imshow(new_depth)
        # plt.savefig(f"{HOME_DIR}/data/{scene_name}/no_table_depth_png/{i:03d}.png")
        np.save(path, depth)
        # np.save(os.path.join(scene_name, "seg_mask", path.split("/")[-1]), mask)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--scene", type=str)
    args = argparser.parse_args()
    scene_name = args.scene
    prime_sphere_main(scene_name)
