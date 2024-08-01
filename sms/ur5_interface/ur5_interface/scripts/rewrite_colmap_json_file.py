import json
import viser
import numpy as np
import viser.transforms as vtf
import copy
from sklearn.neighbors import NearestNeighbors
import os
import open3d as o3d

robot_json_filepath = '/home/lifelong/sms/sms/data/utils/Detic/20240730_paper_bowl2/transforms.json'
colmap_json_filepath = '/home/lifelong/sms/sms/data/utils/Detic/20240730_paper_bowl2/colmap_outputs/transforms.json'
colmap_ply_filepath = colmap_json_filepath[:colmap_json_filepath.rfind('/')] + '/sparse_pc.ply'
pcd = o3d.io.read_point_cloud(colmap_ply_filepath)
points = np.asarray(pcd.points)

def apply_transform_to_pointcloud(pointcloud, transform_matrix):
    """Apply a 4x4 transformation matrix to a point cloud."""
    # Convert point cloud to homogeneous coordinates
    num_points = pointcloud.shape[0]
    homogeneous_points = np.hstack((pointcloud, np.ones((num_points, 1))))
    
    # Apply the transformation
    transformed_points_homogeneous = homogeneous_points @ transform_matrix.T
    
    # Convert back to 3D coordinates
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3][:, np.newaxis]
    
    return transformed_points

def visualize_transform_json(server,data,prefix):
    frame_to_transform_dict = {}
    for frame in data['frames']:
        transform_matrix = np.array(frame['transform_matrix'])
        filename = prefix + '/' + frame['file_path']
        frame_key = int(frame['file_path'][frame['file_path'].find('0')+1:frame['file_path'].find('.')])
        if(prefix == 'robot_pose'):
            frame_key += 1
        frame_to_transform_dict[frame_key] = transform_matrix
        viser_tf = viser.transforms.SE3.from_matrix(transform_matrix)
        server.add_frame(name=filename,wxyz=viser_tf.rotation().wxyz,position=viser_tf.translation(),axes_length=0.05,axes_radius=0.0025)
    return frame_to_transform_dict

def convert_dictionary_to_pointcloud(dictionary):
    pointcloud = []
    for key in dictionary:
        transform_matrix = dictionary[key]
        translation = transform_matrix[:3,3]
        pointcloud.append(translation)
    return np.array(pointcloud)
        
def icp(p_points, q_points):
    # Convert p_points and q_points to numpy arrays
    p_avg = np.mean(p_points,axis=0)
    q_avg = np.mean(q_points,axis=0)
    p_points = p_points - p_avg
    q_points = q_points - q_avg
    x = p_points.T # Shape (3,num_points)
    yt = q_points # Shape (num_points,3)
    s = x @ yt
    U,_,Vt = np.linalg.svd(s)
    V = Vt.T
    print("U matrix:")
    print(U)
    print("V matrix:")
    print(V)
    
    # Compute the rotation matrix R
    R = np.dot(V, U.T)
    print("R matrix:")
    print(R)
    print("Determinant:", np.linalg.det(R))
    
    # Correct for reflection if needed
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = np.dot(V, U.T)
        print("Improved R matrix:")
        print(R)
    
    # Convert averages to numpy arrays
    p_avg_matrix = np.array(p_avg).reshape(3, 1)
    q_avg_matrix = np.array(q_avg).reshape(3, 1)
    
    # Compute the translation vector
    translation = q_avg_matrix - np.dot(R, p_avg_matrix)
    print("Translation:")
    print(translation)
    return np.vstack((np.hstack((R,translation)),np.array([0,0,0,1])))


renamed_robot_json_filepath = robot_json_filepath[:robot_json_filepath.rfind('/')] + '/old_transforms.json'
with open(robot_json_filepath, 'r') as file:
    robot_data = json.load(file)
with open(colmap_json_filepath, 'r') as file:
    colmap_data = json.load(file)
server = viser.ViserServer()
robot_frame_to_transform_dict = visualize_transform_json(server,robot_data,'robot_pose')
colmap_frame_to_transform_dict = visualize_transform_json(server,colmap_data,'colmap')
key_list = sorted(colmap_frame_to_transform_dict.keys())
scales = []
for i in range(len(key_list)):
    print(key_list[i])
    # Inner loop starts from i+1 to avoid redundant comparisons
    for j in range(i + 1, len(key_list)):
        
        robot_tf1 = robot_frame_to_transform_dict[key_list[i]]
        robot_tf2 = robot_frame_to_transform_dict[key_list[j]]
        robot_translation_delta = robot_tf2[:3,3] - robot_tf1[:3,3]
        robot_delta_magnitude = np.linalg.norm(robot_translation_delta)
        
        colmap_tf1 = colmap_frame_to_transform_dict[key_list[i]]
        colmap_tf2 = colmap_frame_to_transform_dict[key_list[j]]
        colmap_translation_delta = colmap_tf2[:3,3] - colmap_tf1[:3,3]
        colmap_delta_magnitude = np.linalg.norm(colmap_translation_delta)
        scale = robot_delta_magnitude / colmap_delta_magnitude
        scales.append(scale)
mean_scale = np.mean(np.array(scales))
scaled_colmap_frame_to_transform_dict =  copy.deepcopy(colmap_frame_to_transform_dict)
for key in scaled_colmap_frame_to_transform_dict:
    transform_matrix = scaled_colmap_frame_to_transform_dict[key]
    translation = transform_matrix[:3,3]
    translation_scaled = translation * mean_scale
    transform_matrix[:3,3] = translation_scaled
    scaled_colmap_frame_to_transform_dict[key] = transform_matrix

# Did it in a separate loop to dummy check that we updating variables out of scope
for key in scaled_colmap_frame_to_transform_dict:
    filename = 'scaled_colmap/' + str(key)
    transform_matrix = scaled_colmap_frame_to_transform_dict[key]
    viser_tf = viser.transforms.SE3.from_matrix(transform_matrix)
    server.add_frame(name=filename,wxyz=viser_tf.rotation().wxyz,position=viser_tf.translation(),axes_length=0.05,axes_radius=0.0025)

scaled_colmap_pc = []
robot_pc = []
    
for key in scaled_colmap_frame_to_transform_dict:
    world_to_scaled_colmap_pos = scaled_colmap_frame_to_transform_dict[key][:3,3]
    scaled_colmap_pc.append(world_to_scaled_colmap_pos)
    world_to_robot_pos = robot_frame_to_transform_dict[key][:3,3]
    robot_pc.append(world_to_robot_pos)
    world_to_scaled_colmap = scaled_colmap_frame_to_transform_dict[key]
    world_to_robot = robot_frame_to_transform_dict[key]
    scaled_colmap_to_robot = np.linalg.inv(world_to_scaled_colmap) @ world_to_robot

scaled_colmap_pc = np.array(scaled_colmap_pc)
robot_pc = np.array(robot_pc)
    
server.add_point_cloud('robot_pc',points=robot_pc,colors=np.zeros((robot_pc.shape[0],3)),point_size=0.005)
server.add_point_cloud('scaled_colmap_pc',points=scaled_colmap_pc,colors=np.zeros((scaled_colmap_pc.shape[0],3)),point_size=0.005)

final_tf = icp(scaled_colmap_pc,robot_pc)

scaled_transformed_colmap_frame_to_transform_dict = copy.deepcopy(scaled_colmap_frame_to_transform_dict)
for key in scaled_transformed_colmap_frame_to_transform_dict:
    world_to_scaled_colmap = scaled_transformed_colmap_frame_to_transform_dict[key]
    world_to_robot = final_tf @ world_to_scaled_colmap
    world_to_robot_position = world_to_robot[:3,3]
    server.add_point_cloud('recovered_pc/'+str(key),world_to_robot_position.reshape(1,-1),np.array([255,0,0]).reshape(1,-1),point_size=0.005)
    
    scaled_transformed_colmap_frame_to_transform_dict[key] = world_to_robot

# Did it in a separate loop to dummy check that we updating variables out of scope
for key in scaled_transformed_colmap_frame_to_transform_dict:
    filename = 'scaled_transformed_colmap/' + str(key)
    transform_matrix = scaled_transformed_colmap_frame_to_transform_dict[key]
    viser_tf = viser.transforms.SE3.from_matrix(transform_matrix)
    server.add_frame(name=filename,wxyz=viser_tf.rotation().wxyz,position=viser_tf.translation(),axes_length=0.05,axes_radius=0.0025)


# vvvvv JY Additions vvvvv
scale_final = mean_scale
tf_final = final_tf # applied after scaling

import pdb
pdb.set_trace()

from pathlib import Path
import os.path as osp
    
colmap_tf_data = copy.deepcopy(colmap_data) # create deepcopy of original colmap json dict to modify tf values
for frame in colmap_tf_data['frames']:
    transform_matrix_old = np.array(frame['transform_matrix'])
    frame['transform_matrix'] = scaled_transformed_colmap_frame_to_transform_dict[frame['colmap_im_id']].tolist()

os.rename(robot_json_filepath,renamed_robot_json_filepath)


json_object = json.dumps(colmap_tf_data, indent=4)
with open(str(robot_json_filepath), "w") as outfile:
    outfile.write(json_object)
print("Visualize tf")
import pdb;pdb.set_trace()

    
