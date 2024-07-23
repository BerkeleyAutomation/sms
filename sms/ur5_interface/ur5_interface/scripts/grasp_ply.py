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
from visualization_utils import visualize_grasps
from autolab_core import RigidTransform
import open3d as o3d
tool_to_wrist = RigidTransform()
# 0.1651 was old measurement is the measure dist from suction to 
# 0.1857375 Parallel Jaw gripper
tool_to_wrist.translation = np.array([0, 0, 0])
tool_to_wrist.from_frame = "tool"
tool_to_wrist.to_frame = "wrist"

robot = UR5Robot(gripper=1)
robot.gripper.open()
robot.set_tcp(tool_to_wrist)
home_joints = np.array([0.3103832006454468, -1.636097256337301, -0.5523660818683069, -2.4390090147601526, 1.524283766746521, 0.29816189408302307])
robot.move_joint(home_joints,vel=1.0,acc=0.1)
    
segmented_ply_filepath = "/home/lifelong/sms/sms/data/utils/Detic/outputs/2024_07_22_green_tape_bowl/prime_seg_gaussians.ply"
full_ply_filepath = "/home/lifelong/sms/sms/data/utils/Detic/outputs/2024_07_22_green_tape_bowl/prime_full_gaussians.ply"
bounding_box_filepath = "/home/lifelong/sms/sms/data/utils/Detic/2024_07_22_green_tape_bowl/table_bounding_cube.json"
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', default=contact_graspnet_path + '/../checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
parser.add_argument('--seg_np_path', default=segmented_ply_filepath, help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
parser.add_argument('--full_np_path', default=full_ply_filepath, help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
parser.add_argument('--pc_bounding_box_path', default=bounding_box_filepath, help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
parser.add_argument('--filter_grasps', action='store_true', default=True,  help='Filter grasp contacts according to segmap.')
parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
parser.add_argument('--forward_passes', type=int, default=10,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
FLAGS = parser.parse_args()

global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)

print(str(global_config))
print('pid: %s'%(str(os.getpid())))

pred_grasps_cam,scores,pc_full,pc_colors = inference(global_config, FLAGS.ckpt_dir, FLAGS.seg_np_path if not FLAGS.png_path else FLAGS.png_path, FLAGS.full_np_path,FLAGS.pc_bounding_box_path, z_range=eval(str(FLAGS.z_range)),
            K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
            forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects,debug=True)

best_scores = {0:scores[0][np.argsort(scores[0])[::-1]][:1]}
best_grasps = {0:pred_grasps_cam[0][np.argsort(scores[0])[::-1]][:1]}
world_to_cam_tf = np.array([[0,-1,0,0],
                                    [-1,0,0,0],
                                    [0,0,-1,0],
                                    [0,0,0,1]])


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
o3d.visualization.draw_geometries([point_cloud_cam,coordinate_frame,grasp_point])

ones = np.ones((pc_full.shape[0],1))
world_to_cam_tf = RigidTransform.load('/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf').matrix
homogenous_points_cam = np.hstack((pc_full,ones))
homogenous_points_world = world_to_cam_tf @ homogenous_points_cam.T
points_world = homogenous_points_world[:3,:] / homogenous_points_world[3,:][np.newaxis,:]
points_world = points_world.T

point_cloud_world = o3d.geometry.PointCloud()

# Set the points and colors
point_cloud_world.points = o3d.utility.Vector3dVector(points_world)
point_cloud_world.colors = o3d.utility.Vector3dVector(pc_colors)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
panda_grasp_point_to_robotiq_grasp_point = np.array([[1,0,0,0],
                                                     [0,1,0,0],
                                                     [0,0,1,-0.06],
                                                     [0,0,0,1]])
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
import pdb
pdb.set_trace()
pre_grasp_rigid_tf = RigidTransform(rotation=pre_grasp_world_frame[:3,:3],translation=pre_grasp_world_frame[:3,3])
robot.move_pose(pre_grasp_rigid_tf,vel=1.0,acc=0.1)
final_grasp_rigid_tf = RigidTransform(rotation=final_grasp_world_frame[:3,:3],translation=final_grasp_world_frame[:3,3])
robot.move_pose(final_grasp_rigid_tf,vel=1.0,acc=0.1)
robot.gripper.close()
robot.move_pose(pre_grasp_rigid_tf,vel=1.0,acc=0.1)
