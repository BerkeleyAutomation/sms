import open3d as o3d
import numpy as np
import viser
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

def filter_noise(points, colors):
    eps = 0.005  # Maximum distance between two samples to be considered as neighbors
    min_samples = 10  # Minimum number of samples in a neighborhood for a point to be a core point

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # Filter out the noise points (label = -1)
    filtered_pointcloud = points[labels != -1]
    filtered_colors = colors[labels != -1]
    return filtered_pointcloud, filtered_colors

def crop_pc(points, colors):
    table_center = np.array([0.48666, -0.0104, -0.120])
    # distances = np.linalg.norm(points - table_center, axis=1)
    # idxs = np.where(distances < 0.5)[0]
    
    x_min_world,x_max_world,y_min_world,y_max_world = 0.258, 0.65, -0.366, 0.3452
    z_min_world,z_max_world = -0.12, 0.134 
    idxs = np.where((points[:, 0] >= x_min_world) & (points[:, 0] <= x_max_world) & (points[:, 1] >= y_min_world) & (points[:, 1] <= y_max_world) & (points[:, 2] >= z_min_world) & (points[:, 2] <= z_max_world))[0]
    points = points[idxs]
    colors = colors[idxs]
    return points, colors

# def isolateTable(points):
# ...

def compute_avg_nn_distance(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    return avg_distance

def voxel_grid_upsample(pcd, voxel_size):
    # Perform voxel downsampling to create a voxel grid
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # Create a dense point cloud by filling the voxel grid
    up_pcd = o3d.geometry.PointCloud()
    up_points = []
    for voxel in down_pcd.get_voxels():
        center = voxel.grid_index * voxel_size
        up_points.append(center)

    up_pcd.points = o3d.utility.Vector3dVector(np.asarray(up_points))
    return up_pcd

def moving_least_squares_upsample(pointcloud, search_radius, polynomial_order):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    # Perform Moving Least Squares (MLS) reconstruction
    mls_pcd = pcd.compute_nearest_neighbor_distance()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    
    mls_pcd = pcd.voxel_down_sample(voxel_size=search_radius)
    return mls_pcd

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--filter_noise", default=False, action="store_true")
    argparser.add_argument("--crop_pc", default=False, action="store_true")
    argparser.add_argument("--type", type=str)
    argparser.add_argument("--data", default=None, type=str)
    args = argparser.parse_args()
    if args.type == "ply":
        ply_filepath = f"{args.data}/sparse_pc.ply"
        pcd = o3d.io.read_point_cloud(ply_filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
    elif args.type == "gs":
        points = np.loadtxt(f"{args.data}/means.txt")
        colors = np.loadtxt(f"{args.data}/colors.txt")#np.random.randint(0, 256, size=points.shape)
        pcd = o3d.geometry.PointCloud()
    elif args.type == "path":
        ply_filepath = args.data
        pcd = o3d.io.read_point_cloud(ply_filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
    if args.crop_pc:
        points, colors = crop_pc(points, colors)
    if args.filter_noise:
        points, colors = filter_noise(points, colors)
    
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # gs pointcloud is too sparse so we need to densify it
    if args.type == "gs":
        # Compute average nearest neighbor distance
        # avg_nn_distance = compute_avg_nn_distance(pcd)
        # # Determine voxel size as a fraction of the average nearest neighbor distance
        # voxel_size = avg_nn_distance * 2 
        # pcd = voxel_grid_upsample(pcd, voxel_size)
        # points = np.array(pcd.points)
        # colors = np.array(pcd.colors)
        pass
    server = viser.ViserServer()
    import pdb
    pdb.set_trace()
    # server.add_point_cloud(name="table_center", points=table_center.reshape((1,3)), colors=np.array([255,0,0]).reshape((1,3)), point_size=0.05)
    server.add_point_cloud(name="pointcloud",points=points,colors=colors,point_size=0.001)
    input("save pointcloud?")
    if not args.type == "path":
        o3d.io.write_point_cloud(f"{args.data}/filtered_{args.type}_pc.ply",pcd)
    input("Kill pointcloud?")
    exit()