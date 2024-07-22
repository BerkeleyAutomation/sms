import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
contact_graspnet_path = os.path.join(dir_path,'../../../contact_graspnet/contact_graspnet')
sys.path.append(contact_graspnet_path)
from prime_inference import inference_points
import argparse
import config_utils
import numpy as np
from ur5py.ur5 import UR5Robot
from visualization_utils import visualize_grasps
from autolab_core import RigidTransform
import open3d as o3d
import math
tool_to_wrist = RigidTransform()
# 0.1651 was old measurement is the measure dist from suction to 
# 0.1857375 Parallel Jaw gripper
tool_to_wrist.translation = np.array([0, 0, 0])
tool_to_wrist.from_frame = "tool"
tool_to_wrist.to_frame = "wrist"

CKPT_DIR = "/home/lifelong/sms/sms/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"
WORLD_TO_CAM_TF = np.array([[0,-1,0,0],
                                [-1,0,0,0],
                                [0,0,-1,0],
                                [0,0,0,1]])
PANDA_GRASP_POINT_TO_ROBOTIQ_GRASP_POINT = np.array([[1,0,0,0],
                                                        [0,1,0,0],
                                                        [0,0,1,-0.06],
                                                        [0,0,0,1]])

_EPS = np.finfo(float).eps * 4.0
def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> np.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (
                1.0 - q[1, 1] - q[2, 2],
                q[0, 1] - q[2, 3],
                q[0, 2] + q[1, 3],
                0.0,
            ),
            (
                q[0, 1] + q[2, 3],
                1.0 - q[0, 0] - q[2, 2],
                q[1, 2] - q[0, 3],
                0.0,
            ),
            (
                q[0, 2] - q[1, 3],
                q[1, 2] + q[0, 3],
                1.0 - q[0, 0] - q[1, 1],
                0.0,
            ),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )

### Gets the grasps for an object pointcloud ###
def get_grasps_obj(points_full, points_segment, colors_full, local_regions, filter_grasps, forward_passes, z_range, debug):
    
    global_config = config_utils.load_config(CKPT_DIR, batch_size=forward_passes)

    pred_grasps_cam,scores,pc_full,pc_colors = inference_points(global_config, CKPT_DIR, points_full, points_segment, colors_full, z_range=eval(str(z_range)),
            local_regions=local_regions, filter_grasps=filter_grasps, segmap_id=0, 
            forward_passes=forward_passes, skip_border_objects=False, debug=False)
    breakpoint()
    all_scores = {-1:scores[-1][np.argsort(scores[-1])[::-1]]}
    all_grasps = {-1:pred_grasps_cam[-1][np.argsort(scores[-1])[::-1]]}
    if debug:
        visualize_grasps(pc_full, all_grasps, all_scores, plot_opencv_cam=True, pc_colors=pc_colors)
        point_cloud_cam = o3d.geometry.PointCloud()

        # Set the points and colors
        point_cloud_cam.points = o3d.utility.Vector3dVector(pc_full)
        point_cloud_cam.colors = o3d.utility.Vector3dVector(pc_colors)
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        grasp_point = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        grasp_point.transform(all_grasps[-1][0])
        o3d.visualization.draw_geometries([point_cloud_cam,coordinate_frame,grasp_point])

    ones = np.ones((pc_full.shape[0],1))
    homogenous_points_cam = np.hstack((pc_full,ones))
    homogenous_points_world = WORLD_TO_CAM_TF @ homogenous_points_cam.T
    points_world = homogenous_points_world[:3,:] / homogenous_points_world[3,:][np.newaxis,:]
    points_world = points_world.T

    if debug:
        point_cloud_world = o3d.geometry.PointCloud()

        # Set the points and colors
        point_cloud_world.points = o3d.utility.Vector3dVector(points_world)
        point_cloud_world.colors = o3d.utility.Vector3dVector(pc_colors)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    final_grasps_tf = []
    final_grasps = []
    for grasp in all_grasps[-1]:
        transformed_grasp = WORLD_TO_CAM_TF @ grasp @ PANDA_GRASP_POINT_TO_ROBOTIQ_GRASP_POINT
        final_grasps_tf.append(transformed_grasp)
        rot, translation = RigidTransform.rotation_and_translation_from_matrix(transformed_grasp)
        rtf = RigidTransform(rotation=rot, translation=translation)
        # NOTE: THIS IS IN Q_WXYZ FORMAT!!!!
        pos, quat = rtf.position, rtf.quaternion
        final_grasp = np.concatenate((pos, quat))
        final_grasps.append(final_grasp)
    
    final_grasps = np.array(final_grasps)
    final_grasps_tf = np.array(final_grasps_tf)
    breakpoint()
    print("CHECK RQ that quaternion_matrix actually works!!!")
    return final_grasps, all_scores
    
def exec_grasp(grasp, points, colors, debug=False):
    if type(grasp) == RigidTransform:
        grasp_tf = grasp.as_matrix()
    elif type(grasp) == tuple:
        # grasp should be in form (pos, quat)
        rot = quaternion_matrix(grasp[1])
        grasp_tf = RigidTransform(rotation=rot,translation=grasp[0]).matrix()
    robot = UR5Robot(gripper=1)
    robot.gripper.open()
    robot.set_tcp(tool_to_wrist)
    breakpoint()
    home_joints = np.array([0.3103832006454468, -1.636097256337301, -0.5523660818683069, -2.4390090147601526, 1.524283766746521, 0.29816189408302307])
    robot.move_joint(home_joints,vel=1.0,acc=0.1)
    pre_grasp_tf = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,-0.1],
                            [0,0,0,1]])
    pre_grasp_world_frame = grasp_tf @ pre_grasp_tf
    if debug:
        grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        grasp_point_world.transform(grasp_tf)
        pre_grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        pre_grasp_point_world.transform(pre_grasp_world_frame)
        o3d.visualization.draw_geometries([point_cloud_world,coordinate_frame,grasp_point_world,pre_grasp_point_world])
    breakpoint()
    
    pre_grasp_rigid_tf = RigidTransform(rotation=pre_grasp_world_frame[:3,:3],translation=pre_grasp_world_frame[:3,3])
    robot.move_pose(pre_grasp_rigid_tf,vel=1.0,acc=0.1)
    final_grasp_rigid_tf = RigidTransform(rotation=grasp_tf[:3,:3],translation=grasp_tf[:3,3])
    robot.move_pose(final_grasp_rigid_tf,vel=1.0,acc=0.1)
    robot.gripper.close()
    robot.move_pose(pre_grasp_rigid_tf,vel=1.0,acc=0.1)
    # kills robot so it doesn't hog the connection
    robot.kill()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=np.ndarray, default=None, help='array of all gaussian means')
    parser.add_argument('--obj_points', type=np.ndarray, default=None, help="array of just the object's gaussian means")
    parser.add_argument('--colors', type=np.ndarray, default=None, help='array of all gaussian colors')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=10,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--debug', type=np.ndarray, default=None, help='Path to the point cloud file')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    FLAGS = parser.parse_args()

    # if not FLAGS.scene:
    #     raise("Provide a scene name")
    
    # global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)

    # print(str(global_config))
    # print('pid: %s'%(str(os.getpid())))
    
    breakpoint()
    # NOTE load in the files of all the things the toad_object would've gotten and
    points = np.loadtxt("/home/lifelong/sms/sms/ur5_interface/ur5_interface/grasp_data/drill_spool/points.txt")
    color = np.loadtxt("/home/lifelong/sms/sms/ur5_interface/ur5_interface/grasp_data/drill_spool/features_dc.txt")
    group_masks = np.loadtxt("/home/lifelong/sms/sms/ur5_interface/ur5_interface/grasp_data/drill_spool/group_masks.txt")
    obj_idx = 0 # 0 for drill, 1 for spool 
    obj_mask = group_masks[obj_idx].astype(np.bool)
    obj_points = points[obj_mask]
    get_grasps_obj(FLAGS.points, FLAGS.obj_points, FLAGS.colors, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, 
                   forward_passes=FLAGS.forward_passes, debug=FLAGS.debug, z_range=FLAGS.z_range)