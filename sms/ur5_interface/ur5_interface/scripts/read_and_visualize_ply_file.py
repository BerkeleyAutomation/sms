import open3d as o3d
import numpy as np
import viser

ply_filepath = "/home/lifelong/sms/sms/data/utils/Detic/outputs/20240818_drill_solo1/sms-data/2024-08-18_172522/global.ply"
pcd = o3d.io.read_point_cloud(ply_filepath)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
# colors[:,[0,2]] = colors[:,[2,0]]
server = viser.ViserServer()
import pdb
pdb.set_trace()
server.add_point_cloud(name="pointcloud",points=points,colors=colors,point_size=0.001)
input("Kill pointcloud?")
exit()