import numpy as np
import open3d as o3d
import viser
import tyro
import time
import json

def get_bbox_from_grasp(grasp, depth=0.1016, width=0.085, height=0.004) -> o3d.geometry.OrientedBoundingBox:
    center = grasp[:3,3]
    rot_matrix = grasp[:3,:3]
    extent=np.array((depth, width, height))
    box = o3d.geometry.OrientedBoundingBox(center,rot_matrix,extent)
    return box

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

# add a func that works with adding a single grasp mesh
# def visualize_grasps()

def visualize_grasps(
    local_ply_filename: str,
    global_ply_filename: str,
    table_bounding_cube_filename: str,
    pred_grasps_filename: str,
    scores_filename: str,
    server
):
    seg_pc = o3d.io.read_point_cloud(local_ply_filename)
    full_pc_unfiltered = o3d.io.read_point_cloud(global_ply_filename)

    full_pc_points = np.asarray(full_pc_unfiltered.points)
    full_pc_colors = np.asarray(full_pc_unfiltered.colors)
    # Crop out noisy Gaussian means
    bounding_box_dict = None
    with open(table_bounding_cube_filename, 'r') as json_file:
        # Step 2: Load the contents of the file into a Python dictionary
        bounding_box_dict = json.load(json_file)
    cropped_indices = (full_pc_points[:, 0] >= bounding_box_dict['x_min']) & (full_pc_points[:, 0] <= bounding_box_dict['x_max']) & (full_pc_points[:, 1] >= bounding_box_dict['y_min']) & (full_pc_points[:, 1] <= bounding_box_dict['y_max']) & (full_pc_points[:, 2] >= bounding_box_dict['z_min']) & (full_pc_points[:, 2] <= bounding_box_dict['z_max'])
    filtered_pc_points = full_pc_points[cropped_indices]
    filtered_pc_colors = full_pc_colors[cropped_indices]
    full_pc = o3d.geometry.PointCloud()
    full_pc.points = o3d.utility.Vector3dVector(filtered_pc_points)
    full_pc.colors = o3d.utility.Vector3dVector(filtered_pc_colors)
    server.add_point_cloud(name="local_pc",points=np.asarray(seg_pc.points),colors=np.asarray(seg_pc.colors),point_size=0.001)
    server.add_point_cloud(name="global_pc",points=filtered_pc_points,colors=filtered_pc_colors,point_size=0.001)
    pred_grasps = np.load(pred_grasps_filename)
    scores = np.load(scores_filename)
    ordered_idxs = np.argsort(scores)[::-1]
    ordered_grasps = pred_grasps[ordered_idxs]
    ordered_scores = scores[ordered_idxs]
    best_score = ordered_scores[0]
    i = 0
    correction_rot = np.array([[0,-1,0], [0,0,-1], [1,0,0]])
    for grasp, score in zip(ordered_grasps, ordered_scores):
        center = grasp[:3, 3]
        rot_matrix = grasp[:3,:3]@correction_rot
        # depth is how long to make the prongs, robotiq gripper width should be 85mm
        grasp_mesh, grasp_color = plot_gripper_pro_max(center=center, R=rot_matrix, width=0.085, depth=0.1016, score=score)
        if score == best_score:
            # we have the best grasp be shown in blue
            grasp_color = [0,1,0]
        server.add_mesh_simple(
                name=f"/grasp_{i}/mesh",
                vertices=np.asarray(grasp_mesh.vertices),
                faces=np.asarray(grasp_mesh.triangles),
                color=grasp_color
            )
        i += 1
    # breakpoint()
    
def main():
    data_dir = "/home/lifelong/sms/sms/data/utils/Detic/outputs/0808_drill_battery_nofeatup/sms-data/2024-08-09_113806"
    local_ply_filename = data_dir+"/local.ply"
    global_ply_filename = data_dir+"/global.ply"
    table_bounding_cube_filename = "/home/lifelong/sms/sms/data/utils/Detic/0808_drill_battery_nofeatup/table_bounding_cube.json"
    pred_grasps_filename = data_dir+"/pred_grasps_world.npy"
    scores_filename = data_dir+"/scores.npy"
    server = viser.ViserServer()
    visualize_grasps(local_ply_filename, global_ply_filename, table_bounding_cube_filename, pred_grasps_filename, scores_filename, server)
    
if __name__ == "__main__":
    tyro.cli(main)