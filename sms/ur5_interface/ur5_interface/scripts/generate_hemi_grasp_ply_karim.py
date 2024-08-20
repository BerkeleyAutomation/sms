import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
contact_graspnet_path = os.path.join(dir_path,'../../../contact_graspnet/contact_graspnet')
sys.path.append(contact_graspnet_path)
from prime_inference import inference
import argparse
import config_utils
import numpy as np
from ur5py.ur5 import UR5Robot
import matplotlib.pyplot as plt 
from visualization_utils import visualize_grasps
from autolab_core import RigidTransform
import open3d as o3d
tool_to_wrist = RigidTransform()
# 0.1651 was old measurement is the measure dist from suction to 
# 0.1857375 Parallel Jaw gripper
tool_to_wrist.translation = np.array([0, 0, 0])
tool_to_wrist.from_frame = "tool"
tool_to_wrist.to_frame = "wrist"
    
segmented_ply_filepath = "/home/lifelong/sms/sms/data/utils/Detic/outputs/2024_07_22_green_tape_bowl/prime_seg_gaussians.ply"
full_ply_filepath = "/home/lifelong/sms/sms/data/utils/Detic/outputs/2024_07_22_green_tape_bowl/prime_full_gaussians.ply"
bounding_box_filepath = "/home/lifelong/sms/sms/data/utils/Detic/2024_07_22_green_tape_bowl/table_bounding_cube.json"

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

def generate_hemi_grasps(seg_np_path, full_np_path, pc_bounding_box_path, ckpt_dir, z_range, K, local_regions, filter_grasps, skip_border_objects, forward_passes, segmap_id, arg_configs, save_dir):
    # angle range from top of sphere
    phi_min, phi_max = 90, 20
    # phi_min, phi_max = 180, -180
    theta_min, theta_max = 180, -180 #125, -125
    phi_div, theta_div = 4, 12
    object_center = np.array([0.48666, -0.0104, -0.120]) 
    radius = 0.35
    
    translations = get_hemi_translations(
        phi_min, phi_max, theta_min, theta_max, object_center, phi_div, theta_div, radius
    )
    rotations = [point_at(translation, object_center) for translation in translations]
    poses = [
        RigidTransform(rotations[i], translations[i], from_frame="cam")
        # * wrist_to_cam.inverse()
        for i in range(len(translations))
    ]
    visualize_poses(poses)
    all_pred_grasps = []
    all_scores = []
    all_contact_pts = []
    for cam_pose in poses:
        breakpoint()
        cam_pose_matrix = cam_pose.matrix
        cam_pose_matrix[np.abs(cam_pose_matrix) < 1e-15] = 0
        contact_graspnet_env_path = "/home/lifelong/anaconda3/envs/contact_graspnet/bin/python"
        generate_grasps_path = "/home/lifelong/sms/sms/ur5_interface/ur5_interface/scripts/generate_grasp_ply_karim.py"
        print(contact_graspnet_env_path + " " + generate_grasps_path + ' --seg_np_path ' + seg_np_path + ' --full_np_path ' + full_np_path + ' --pc_bounding_box_path ' + pc_bounding_box_path + ' --save_dir ' + save_dir)
        result = subprocess.run([contact_graspnet_env_path, generate_grasps_path, "--seg_np_path", seg_np_path, "--full_np_path", full_np_path, "--pc_bounding_box_path", pc_bounding_box_path, "--save_dir", save_dir], capture_output=True, text=True)
        pred_grasps, scores, contact_pts = generate_grasps(seg_np_path, full_np_path, pc_bounding_box_path, ckpt_dir, z_range, K, local_regions, filter_grasps, skip_border_objects, forward_passes, segmap_id, arg_configs, save_dir, cam_pose_matrix)
        all_pred_grasps.append(pred_grasps)
        all_scores.append(scores)
        all_contact_pts.append(contact_pts)
    # WANT TO DO FILTERING HERE TO NOT HAVE SO MANY GRASPS
    np.save(f'{FLAGS.save_dir}/pred_grasps_world.npy', all_pred_grasps)
    np.save(f'{FLAGS.save_dir}/scores.npy', all_scores)
    np.save(f'{FLAGS.save_dir}/contact_pts.npy', all_contact_pts)
    breakpoint()
    
    
    
def generate_grasps(seg_np_path, full_np_path, pc_bounding_box_path, ckpt_dir, z_range, K, local_regions, filter_grasps, skip_border_objects, forward_passes, segmap_id, arg_configs, save_dir, cam_pose=None):

    global_config = config_utils.load_config(ckpt_dir, batch_size=forward_passes, arg_configs=arg_configs)

    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))
    
    if cam_pose is None:
        # world_to_cam_tf = np.array([[0,-1,0,0],
        #                             [-1,0,0,0],
        #                             [0,0,-1,0],
        #                             [0,0,0,1]])
        world_to_cam_tf = RigidTransform.load('/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed_for_grasping.tf').matrix
    else:
        world_to_cam_tf = cam_pose
        
    pred_grasps_cam, scores, contact_pts, pc_full, pc_colors = inference(global_config, ckpt_dir, seg_np_path, full_np_path,pc_bounding_box_path, z_range=z_range,
                K=K, local_regions=local_regions, filter_grasps=filter_grasps, segmap_id=segmap_id, 
                forward_passes=forward_passes, skip_border_objects=skip_border_objects,debug=True, world_to_cam_tf=world_to_cam_tf)
    
    sorted_idxs = np.argsort(scores[0])[::-1]
    best_scores = {0:scores[0][sorted_idxs][:1]}
    best_grasps = {0:pred_grasps_cam[0][sorted_idxs][:1]}
    best_contact_pts = {0:contact_pts[0][sorted_idxs][:1]}
    
    visualize_grasps(pc_full, best_grasps, best_scores, plot_opencv_cam=True, pc_colors=pc_colors)
    # Create an Open3D point cloud object
    point_cloud_cam = o3d.geometry.PointCloud()

    # Set the points and colors
    point_cloud_cam.points = o3d.utility.Vector3dVector(pc_full)
    point_cloud_cam.colors = o3d.utility.Vector3dVector(pc_colors)

    # Step 2: Visualize the point cloud
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    grasp_point = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    grasp_point.transform(best_grasps[0][0])
    # o3d.visualization.draw_geometries([point_cloud_cam,coordinate_frame,grasp_point])

    ones = np.ones((pc_full.shape[0],1))
    # world_to_cam_tf = RigidTransform.load('/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed_for_grasping.tf').matrix
    homogenous_points_cam = np.hstack((pc_full,ones))
    homogenous_points_world = world_to_cam_tf @ homogenous_points_cam.T
    points_world = homogenous_points_world[:3,:] / homogenous_points_world[3,:][np.newaxis,:]
    points_world = points_world.T

    point_cloud_world = o3d.geometry.PointCloud()

    # Set the points and colors
    point_cloud_world.points = o3d.utility.Vector3dVector(points_world)
    point_cloud_world.colors = o3d.utility.Vector3dVector(pc_colors)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    panda_grasp_point_to_robotiq_grasp_point = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-0.02],[0,0,0,1]]) # -0.06
    final_grasp_world_frame = world_to_cam_tf @ best_grasps[0][0] @ panda_grasp_point_to_robotiq_grasp_point
    grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    grasp_point_world.transform(final_grasp_world_frame)
    pre_grasp_tf = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,-0.1],
                            [0,0,0,1]])
    pre_grasp_world_frame = final_grasp_world_frame @ pre_grasp_tf
    pre_grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    pre_grasp_point_world.transform(pre_grasp_world_frame)
    o3d.visualization.draw_geometries([point_cloud_world,coordinate_frame,grasp_point_world,pre_grasp_point_world])
    pred_grasps_world = []
    for i in range(len(pred_grasps_cam[0])):
        grasp = world_to_cam_tf @ pred_grasps_cam[0][i] @ panda_grasp_point_to_robotiq_grasp_point
        pred_grasps_world.append(grasp)
    pred_grasps_world = np.array(pred_grasps_world)
    
    # only want to save when we're not using the hemisphere
    # if cam_pose is None:
    #     np.save(f'{FLAGS.save_dir}/pred_grasps_world.npy', pred_grasps_world)
    #     np.save(f'{FLAGS.save_dir}/scores.npy', scores[0])
    #     np.save(f'{FLAGS.save_dir}/contact_pts.npy', contact_pts[0])
    
    return pred_grasps_world, scores[0], contact_pts[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_np_path', default=segmented_ply_filepath)
    parser.add_argument('--full_np_path', default=full_ply_filepath)
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--ckpt_dir', default='/home/lifelong/sms/sms/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--pc_bounding_box_path', default=bounding_box_filepath, help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=None, help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=True,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=10,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()
    # generate_grasps(FLAGS.seg_np_path, FLAGS.full_np_path, FLAGS.pc_bounding_box_path, FLAGS.ckpt_dir, FLAGS.z_range, FLAGS.K, FLAGS.local_regions, 
                    # FLAGS.filter_grasps, FLAGS.skip_border_objects, FLAGS.forward_passes, FLAGS.segmap_id, FLAGS.arg_configs, FLAGS.save_dir)
    generate_hemi_grasps(FLAGS.seg_np_path, FLAGS.full_np_path, FLAGS.pc_bounding_box_path, FLAGS.ckpt_dir, FLAGS.z_range, FLAGS.K, FLAGS.local_regions, 
                    FLAGS.filter_grasps, FLAGS.skip_border_objects, FLAGS.forward_passes, FLAGS.segmap_id, FLAGS.arg_configs, FLAGS.save_dir)