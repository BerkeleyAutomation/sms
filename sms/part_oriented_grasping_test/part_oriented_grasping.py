import numpy as np
import viser
import open3d as o3d
from sms.tracking.grasp_vis_utils import visualize_grasps

def filter_top_10_percent(point_cloud, reference_point):
    # Step 1: Compute the Euclidean distance between each point in the cloud and the reference point
    distances = np.linalg.norm(point_cloud - reference_point, axis=1)

    # Step 2: Sort points by distance (ascending)
    sorted_indices = np.argsort(distances)

    # Step 3: Select the top 10% of points
    top_10_percent_count = int(0.1 * len(point_cloud))
    
    # Get the indices of the top 10% closest points
    top_10_percent_indices = sorted_indices[:top_10_percent_count]

    return top_10_percent_indices

def cosine_similarity_matrix(dino_features_for_object, top_dino):
    # Step 1: Normalize the dino_features_for_object (Nx64) and top_dino (64,)
    dino_features_norm = dino_features_for_object / np.linalg.norm(dino_features_for_object, axis=1, keepdims=True)
    top_dino_norm = top_dino / np.linalg.norm(top_dino)
    
    # Step 2: Calculate the cosine similarity (dot product) between each normalized vector and top_dino
    cosine_similarities = np.dot(dino_features_norm, top_dino_norm)
    
    return cosine_similarities

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
        ''' Author: chenxi-wang
        Create box instance with mesh representation.
        '''
        box = o3d.geometry.TriangleMesh()
        vertices = np.array([[0,0,0],
                            [width,0,0],
                            [0,0,depth],
                            [width,0,depth],
                            [0,height,0],
                            [width,height,0],
                            [0,height,depth],
                            [width,height,depth]])
        vertices[:,0] += dx
        vertices[:,1] += dy
        vertices[:,2] += dz
        triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                            [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                            [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
        box.vertices = o3d.utility.Vector3dVector(vertices)
        box.triangles = o3d.utility.Vector3iVector(triangles)
        return box

def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
    '''
    Author: chenxi-wang
    
    **Input:**

    - center: numpy array of (3,), target point as gripper center

    - R: numpy array of (3,3), rotation matrix of gripper

    - width: float, gripper width

    - score: float, grasp quality score

    **Output:**

    - open3d.geometry.TriangleMesh
    '''
    x, y, z = center
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    
    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = 1 - score # red for low score
        color_g = score # green for high score
        color_b = 0 
    
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper, colors[0]

def viser_grasps(points,colors,pred_grasps_filepath, scores_filepath, server,top_dino):
    pred_grasps = np.load(pred_grasps_filepath)
    scores = np.load(scores_filepath)
    ordered_idxs = np.argsort(scores)[::-1]
    ordered_grasps = pred_grasps[ordered_idxs]
    ordered_scores = scores[ordered_idxs]
    semantic_scores = []
    best_score = ordered_scores[0]
    i = 0
    correction_rot = np.array([[0,-1,0], [0,0,-1], [1,0,0]])
    for grasp, score in zip(ordered_grasps, ordered_scores):
        center = grasp[:3, 3]
        rot_matrix = grasp[:3,:3]@correction_rot
        # depth is how long to make the prongs, robotiq gripper width should be 85mm
        # grasp_mesh, grasp_color = plot_gripper_pro_max(center=center, R=rot_matrix, width=0.085, depth=0.1016, score=score)
        plot_gripper_matrix = np.eye(4)
        plot_gripper_matrix[:3,:3] = rot_matrix
        plot_gripper_matrix[:3,3] = center
        contact_point_offset = np.array([[1,0,0,0.1016 + (0.022/2)],
                                         [0,1,0,0],
                                         [0,0,1,0],
                                         [0,0,0,1]])
        new_contact_point_matrix = plot_gripper_matrix @ contact_point_offset
        center = new_contact_point_matrix[:3,3]
        grasp_box_mask = filter_points_in_bounding_box(points,center,rot_matrix,np.array([0.022,0.085,0.038]))
        dinos_in_box = dino_features_for_object[grasp_box_mask]
        dino_scores_in_box = cosine_similarity_matrix(dinos_in_box, top_dino)
        new_grasp_score = np.mean(dino_scores_in_box)
        semantic_scores.append(new_grasp_score)
        i += 1
    semantic_scores = np.array(semantic_scores)
    normalized_semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
    i = 0
    for grasp, score in zip(ordered_grasps, normalized_semantic_scores):
        center = grasp[:3, 3]
        rot_matrix = grasp[:3,:3]@correction_rot
        grasp_mesh, grasp_color = plot_gripper_pro_max(center=center, R=rot_matrix, width=0.085, depth=0.1016, score=score)
        server.add_mesh_simple(
                name=f"/grasp_color/{i}/mesh",
                vertices=np.asarray(grasp_mesh.vertices),
                faces=np.asarray(grasp_mesh.triangles),
                color=grasp_color
            )
        server.add_mesh_simple(
                name=f"/grasp_no_color/{i}/mesh",
                vertices=np.asarray(grasp_mesh.vertices),
                faces=np.asarray(grasp_mesh.triangles),
                color=np.array([0,0,0]).reshape(3,)
            )
        i += 1
    max_grasp = ordered_grasps[np.argmax(normalized_semantic_scores)]
    max_center = max_grasp[:3, 3]
    max_rot_matrix = max_grasp[:3,:3]@correction_rot
    max_grasp_mesh, max_grasp_color = plot_gripper_pro_max(center=max_center, R=max_rot_matrix, width=0.085, depth=0.1016, score=np.max(normalized_semantic_scores))
    server.add_mesh_simple(
                name="max_grasp",
                vertices=np.asarray(max_grasp_mesh.vertices),
                faces=np.asarray(max_grasp_mesh.triangles),
                color=max_grasp_color
            )
    
        

def filter_points_in_bounding_box(point_cloud, bbox_center, rotation_matrix, dimensions):
    # Step 1: Translate the point cloud by the center of the bounding box
    translated_point_cloud = point_cloud - bbox_center
    
    # Step 2: Rotate the point cloud by the inverse of the rotation matrix
    # The inverse of a rotation matrix is its transpose
    rotated_point_cloud = np.dot(translated_point_cloud, rotation_matrix)
    
    # Step 3: Check which points are within the bounding box dimensions
    half_dimensions = dimensions / 2.0
    mask = np.all(np.abs(rotated_point_cloud) <= half_dimensions, axis=1)

    return mask

local_ply_filepath = '/home/lifelong/sms/sms/part_oriented_grasping_test/local.ply'
global_ply_filepath = '/home/lifelong/sms/sms/part_oriented_grasping_test/global.ply'
table_bounding_cube_filepath = '/home/lifelong/sms/sms/part_oriented_grasping_test/table_bounding_cube.json'
part_relevancies_filepath = '/home/lifelong/sms/sms/part_oriented_grasping_test/part_relevancies.npy'
dino_filepath = '/home/lifelong/sms/sms/part_oriented_grasping_test/dino_features_for_object.npy'
pred_grasps_filepath = '/home/lifelong/sms/sms/part_oriented_grasping_test/pred_grasps_world.npy'
scores_filepath = '/home/lifelong/sms/sms/part_oriented_grasping_test/scores.npy'

pcd = o3d.io.read_point_cloud(local_ply_filepath)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

part_relevancies = np.load(part_relevancies_filepath)
dino_features_for_object = np.load(dino_filepath)

# Normalize the part_relevancies to be between 0 and 1
normalized_relevancies = (part_relevancies - part_relevancies.min()) / (part_relevancies.max() - part_relevancies.min())

# Define colors for red and green
red = np.array([255, 0, 0])  # RGB for red
green = np.array([0, 255, 0])  # RGB for green

# Interpolate between red and green based on the normalized relevancies
relevancy_colors = np.outer(1 - normalized_relevancies, red) + np.outer(normalized_relevancies, green)
relevancy_colors = np.round(relevancy_colors).astype(np.uint8)

# Find the thresholds for the 70th and 80th percentiles
lower_threshold = np.percentile(part_relevancies, 35)
upper_threshold = np.percentile(part_relevancies, 35.05)

# Filter points and part_relevancies based on the thresholds
mask = (part_relevancies >= lower_threshold) & (part_relevancies < upper_threshold)
mask = mask.reshape(-1,)

# Extract the top 10% of points and their corresponding scores
top_points = points[mask]
top_colors = colors[mask]
top_dinos = dino_features_for_object[mask]
min_z_index = np.argmin(top_points[:, 2])
top_point = top_points[min_z_index].reshape(-1,3)

top_point_10_percent_mask = filter_top_10_percent(points,top_point)
top_point_10_percent = points[top_point_10_percent_mask]
top_color_10_percent = colors[top_point_10_percent_mask]
top_dino_10_percent = dino_features_for_object[top_point_10_percent_mask]

max_x_index = np.argmax(top_point_10_percent[:,0])
top_point = top_point_10_percent[max_x_index].reshape(-1,3)
top_color = np.array([0,255,0]).reshape(-1,3)
top_dino = top_dino_10_percent[max_x_index]
dino_scores = cosine_similarity_matrix(dino_features_for_object, top_dino)
normalized_dino_scores = (dino_scores - dino_scores.min()) / (dino_scores.max() - dino_scores.min())
# Interpolate between red and green based on the normalized relevancies
dino_colors = np.outer(1 - normalized_dino_scores, red) + np.outer(normalized_dino_scores, green)
dino_colors = np.round(dino_colors).astype(np.uint8)

server = viser.ViserServer()
server.add_point_cloud(name="original_pointcloud",points=points,colors=colors,point_size=0.001)
server.add_point_cloud(name="relevancy_pointcloud",points=points,colors=relevancy_colors,point_size=0.001)
server.add_point_cloud(name="top_relevancy_pointcloud",points=top_point,colors=top_color,point_size=0.01,point_shape='sparkle')
server.add_point_cloud(name="dino_pointcloud",points=points,colors=dino_colors,point_size=0.001)
viser_grasps(points,colors,pred_grasps_filepath, scores_filepath, server,top_dino)
import pdb
pdb.set_trace()