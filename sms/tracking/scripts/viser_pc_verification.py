import viser
import numpy as np
import open3d as o3d
from autolab_core import RigidTransform
from plyfile import PlyData
import time

from pathlib import Path
import tyro

def transform_points(points, ids, tfs, traj0):
    unique_ids = tfs.keys()
    new_points = np.zeros_like(points)
    for no, id in enumerate(unique_ids):
        #     rigid_transform_mat = frame[i].as_matrix()
        #     rigid_transform_mat[:3,3] = rigid_transform_mat[:3,3] - xyz0[i].translation()
        #     # print(f"Object {i}: ", rigid_transform_mat)
        #     means_centered = torch.subtract(self.init_means[mask], self.init_means[mask].mean(dim=0))
        #     means_centered_homog = torch.cat([means_centered, torch.ones(means_centered.shape[0], 1).to(self.device)], dim=1)
        #     # import pdb; pdb.set_trace()
        #     self.model.gauss_params["means"][mask] = ((torch.from_numpy(rigid_transform_mat).to(torch.float32).cuda() @ means_centered_homog.T).T)[:, :3] + self.init_means[mask].mean(dim=0)
        subpoints_ind = np.asarray(ids == id).nonzero()
        subpoints = points[subpoints_ind]
        subpoints_mean = np.mean(subpoints, axis=0)
        subpoints = subpoints - subpoints_mean
        
        rgtf = tfs[id]
        rgtf[:3,3] = rgtf[:3,3] - traj0[no].translation()
        
        ones = np.ones((subpoints.shape[0],1))
        homogenous_points_cam = np.hstack((subpoints,ones))
        subpoints_new = tfs[id] @ homogenous_points_cam.T
        #subpoints_new = homogenous_points_world[:3,:] / homogenous_points_world[3,:][np.newaxis,:]
        subpoints_new = subpoints_new.T[:, :3] + subpoints_mean
        
        new_points[subpoints_ind] = subpoints_new
    
    return new_points

def main(ply_path : tuple[Path, ...], traj_path : tuple[Path, ...]):
    server = viser.ViserServer()
    gui_reset_up = server.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )
    
    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = RigidTransform.rotation_from_quaternion(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
    )
    
    SH_C0 = 0.28209479177387814
    
    data = PlyData.read(ply_path[0])
    v = data["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    
    cluster_path = ply_path[0].parent.parent.parent.joinpath("clusters_2.npy")
    clusters = np.load(cluster_path, allow_pickle=True)
    
    labels = clusters[0]  
    cluster_gaussians_no = []
    ids = []
    for group_no in range(1, len(clusters)):
        cluster_gaussians_no.extend(clusters[group_no])
        
        for point in clusters[group_no]:
            ids.append(int(labels[point].item()))
    
    new_points = positions[cluster_gaussians_no]
    new_colors = colors[cluster_gaussians_no]
    ids = np.array(ids)
    
    server.add_point_cloud('pc',points=new_points,colors=new_colors,point_size=0.001)
    
    ### loop through trajectories and generate a new pointcloud for each one ###
    final_tf = dict()
    trajectories = np.load(traj_path[0], allow_pickle=True)
    traj0 = trajectories[0]
    curr_points, curr_colors = new_points, new_colors
    unique_clusters = list(set(ids))
    while True:
        
        server.add_point_cloud(f'pc',points=new_colors,colors=new_colors,point_size=0.001)
        
        for ind, traj in enumerate(trajectories):
            
            for i, label in enumerate(unique_clusters):
                quat, pos = traj[i].rotation(), traj[i].translation()
                rotation = RigidTransform.rotation_from_quaternion(quat.wxyz)
                rtf = RigidTransform(rotation=rotation, translation=pos)
                
                final_tf[label] = rtf.matrix
                # if not label in final_tf:
                #     final_tf[label] = rtf.matrix
                #     print(f"before {label}: {rtf.matrix}")
                # else:
                #     final_tf[label] = final_tf[label] @ rtf.matrix
            
            #import pdb; pdb.set_trace()
            curr_points = transform_points(new_points, ids, final_tf, traj0)
            
            #import pdb; pdb.set_trace()
            server.add_point_cloud(f'pc',points=curr_points,colors=new_colors,point_size=0.001)
            
            time.sleep(0.25)
            
            #server.reset()

        time.sleep(1.0)
    
    input("Kill Pointcloud?")
    return 1

if __name__ == "__main__":
    tyro.cli(main)