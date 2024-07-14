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

HOME_DIR = "/home/lifelong/sms/sms/ur5_interface/ur5_interface"
# wrist_to_cam = RigidTransform.load("/home/lifelong/ur5_legs/T_webcam_wrist.tf")
wrist_to_cam = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
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

def prime_sphere_main(scene_name, single_image=False, flip_table=False):
    save_dirs = set_up_dirs(scene_name)
    use_robot, use_cam = True, True
    if use_robot:
        robot = UR5Robot(gripper=1)
        clear_tcp(robot)
        home_joints = np.array([0.007606238126754761, -1.5527289549456995, -1.573484245930807, -1.599767033253805, 1.610335350036621, 0.020352210849523544])
        # # safe if moving from suction home pose
        robot.move_joint(home_joints)
        robot.gripper.open()
    if use_cam:
        cam = Zed()
    # pdb.set_trace()
    tool_to_wrist = RigidTransform()
    # need to set to zero so the frame is at the wrist joint
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)

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
            cam_pose = save_pose(wrist_pose, i, save_dirs["poses"])
            world_to_image_frame = cam_pose @ nerf_frame_to_image_frame
            world_to_image_rigid_tf = RigidTransform(rotation=world_to_image_frame[:3,:3],translation=world_to_image_frame[:3,3],from_frame="image_frame_"+str(i),to_frame="world")
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
    i = 0
    
    if use_cam:
        camera_intr = cam.get_intr()
        camera_intr.save(f"{HOME_DIR}/data/{scene_name}/zed.intr")
        save_poses(save_dirs["poses"], cam.get_ns_intrinsics(), wrist_to_cam)
    global_pointcloud = None
    global_rgbcloud = None
    for (left_image,right_image,world_to_image) in zip(left_images,right_images,world_to_images):
        depth,points,rgbs  = cam.get_depth_image_and_pointcloud(left_image,right_image,from_frame="image_frame_"+str(i))
        pointcloud_world_frame = world_to_image.apply(points)
        if(global_pointcloud is None):
            global_pointcloud = pointcloud_world_frame.data.T
            global_rgbcloud = rgbs.data.T
        else:
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
    distances = np.linalg.norm(global_pointcloud - table_center,axis=1)
    threshold_distance = 1.0
    close_pointcloud = global_pointcloud[distances <= threshold_distance]
    close_rgbcloud = global_rgbcloud[distances <= threshold_distance]
    num_gaussians_initialization = 200000
    gaussian_indices = np.random.choice(close_pointcloud.shape[0],num_gaussians_initialization,replace=False)
    subsampled_pointcloud = close_pointcloud[gaussian_indices]
    subsampled_rgbcloud = close_rgbcloud[gaussian_indices]
    server = viser.ViserServer()
    server.add_point_cloud(name="pointcloud",points=subsampled_pointcloud,colors=subsampled_rgbcloud,point_size=0.001)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(subsampled_pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(subsampled_rgbcloud / 255.)
    o3d.io.write_point_cloud(os.path.join(save_dirs['poses'],'..','sparse_pc.ply'),pcd)
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
