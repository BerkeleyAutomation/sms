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

HOME_DIR = "/home/lifelong/sms/sms/ur5_interface/ur5_interface"
# wrist_to_cam = RigidTransform.load("/home/lifelong/ur5_legs/T_webcam_wrist.tf")
wrist_to_cam = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
# threshold to filte
nerf_frame_to_image_frame = np.array([[1,0,0,0],
                                        [0,-1,0,0],
                                        [0,0,-1,0],
                                        [0,0,0,1]])

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
if __name__ == "__main__":
    robot = UR5Robot(gripper=1)
    clear_tcp(robot)
    
    home_joints = np.array([0.30947089195251465, -1.2793572584735315, -2.035713497792379, -1.388848606740133, 1.5713528394699097, 0.34230729937553406])
    robot.move_joint(home_joints,vel=1.0,acc=0.1)
    world_to_wrist = robot.get_pose()
    world_to_wrist.from_frame = "wrist"
    world_to_cam = world_to_wrist * wrist_to_cam
    proper_world_to_cam_translation = world_to_cam.translation
    proper_world_to_cam_rotation = np.array([[0,1,0],[1,0,0],[0,0,-1]])
    proper_world_to_cam = RigidTransform(rotation=proper_world_to_cam_rotation,translation=proper_world_to_cam_translation,from_frame='cam',to_frame='world')
    proper_world_to_wrist = proper_world_to_cam * wrist_to_cam.inverse()
    
    robot.move_pose(proper_world_to_wrist,vel=1.0,acc=0.1)
    import pdb
    pdb.set_trace()
    zed_mini_focal_length = 730
    cam = Zed()
    if(abs(cam.f_ - zed_mini_focal_length) > 10):
        print("Accidentally connected to wrong Zed. Trying again")
        cam = Zed()
        if(abs(cam.f_ - zed_mini_focal_length) > 10):
            print("Make sure just Zed mini is plugged in")
            exit()
    img_l, img_r = cam.get_rgb()
    depth,points,rgbs  = cam.get_depth_image_and_pointcloud(img_l,img_r,from_frame="cam")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.data.T)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    table_points = points.data.T[inliers]
    # These were manually tuned
    db = DBSCAN(eps=0.01, min_samples=600).fit(table_points)
    filtered_table_point_mask = (db.labels_ != -1)
    filtered_table_pointcloud = points.data.T[inliers][filtered_table_point_mask]
    min_bounding_cube_camera_frame = np.array([np.min(filtered_table_pointcloud[:,0]),np.min(filtered_table_pointcloud[:,1]),np.min(points.data.T[:,2]),1]).reshape(-1,1)
    max_bounding_cube_camera_frame = np.array([np.max(filtered_table_pointcloud[:,0]),np.max(filtered_table_pointcloud[:,1]),np.max(filtered_table_pointcloud[:,2]),1]).reshape(-1,1)
    x_min_cam = min_bounding_cube_camera_frame[0,0]
    y_min_cam = min_bounding_cube_camera_frame[1,0]
    z_min_cam = min_bounding_cube_camera_frame[2,0]
    x_max_cam = max_bounding_cube_camera_frame[0,0]
    y_max_cam = max_bounding_cube_camera_frame[1,0]
    z_max_cam = max_bounding_cube_camera_frame[2,0]
    full_filtered_pointcloud = points.data.T[
        (points.data.T[:, 0] >= x_min_cam) & (points.data.T[:, 0] <= x_max_cam) &
        (points.data.T[:, 1] >= y_min_cam) & (points.data.T[:, 1] <= y_max_cam) &
        (points.data.T[:, 2] >= z_min_cam) & (points.data.T[:, 2] <= z_max_cam)
    ]
    full_filtered_rgbcloud = rgbs.data.T[
        (points.data.T[:, 0] >= x_min_cam) & (points.data.T[:, 0] <= x_max_cam) &
        (points.data.T[:, 1] >= y_min_cam) & (points.data.T[:, 1] <= y_max_cam) &
        (points.data.T[:, 2] >= z_min_cam) & (points.data.T[:, 2] <= z_max_cam)
    ]
    
    print("Check if you are connected to Viser")
    import pdb
    pdb.set_trace()
    server = viser.ViserServer()
    
    server.add_point_cloud(name="pointcloud",points=points.data.T,colors=rgbs.data.T,point_size=0.001)
    server.add_point_cloud(name="pointcloud_table",points=points.data.T[inliers],colors=rgbs.data.T[inliers],point_size=0.001)
    server.add_point_cloud(name="pointcloud_table_filtered",points=points.data.T[inliers][filtered_table_point_mask],colors=rgbs.data.T[inliers][filtered_table_point_mask],point_size=0.001)
    server.add_point_cloud(name="full_pointcloud_filtered",points=full_filtered_pointcloud,colors=full_filtered_rgbcloud,point_size=0.001)
    
    import pdb
    pdb.set_trace()
    points_world_frame = proper_world_to_cam.apply(points)
    
    import pdb
    pdb.set_trace()
    
    server = viser.ViserServer()
    
    server.add_point_cloud(name="pointcloud_world",points=points_world_frame.data.T,colors=rgbs.data.T,point_size=0.001)
    
    import pdb
    pdb.set_trace()
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
    
    
    full_filtered_pointcloud_world = points_world_frame.data.T[(points_world_frame.data.T[:, 0] >= x_min_world) & (points_world_frame.data.T[:, 0] <= x_max_world) &
        (points_world_frame.data.T[:, 1] >= y_min_world) & (points_world_frame.data.T[:, 1] <= y_max_world) &
        (points_world_frame.data.T[:, 2] >= z_min_world) & (points_world_frame.data.T[:, 2] <= z_max_world)
    ]
    full_filtered_rgbcloud_world = rgbs.data.T[
        (points_world_frame.data.T[:, 0] >= x_min_world) & (points_world_frame.data.T[:, 0] <= x_max_world) &
        (points_world_frame.data.T[:, 1] >= y_min_world) & (points_world_frame.data.T[:, 1] <= y_max_world) &
        (points_world_frame.data.T[:, 2] >= z_min_world) & (points_world_frame.data.T[:, 2] <= z_max_world)
    ]
    
    import pdb
    pdb.set_trace()
    
    server = viser.ViserServer()
    
    server.add_point_cloud(name="pointcloud_world",points=points_world_frame.data.T,colors=rgbs.data.T,point_size=0.001)
    server.add_point_cloud(name="full_pointcloud_world_filtered",points=full_filtered_pointcloud_world,colors=full_filtered_rgbcloud_world,point_size=0.001)
    input("Kill pointcloud?")