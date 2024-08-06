"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict, Optional

import typer
from typing_extensions import Annotated

import numpy as onp
import numpy.typing as onpt
import tyro
from plyfile import PlyData

from autolab_core import RigidTransform

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
viser_main_path = os.path.join(dir_path,'viser_splatting/viser/')
sys.path.append(viser_main_path)

import src.viser as vsplat


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""
    ids: dict
    """(N : {0,...,no_clusters-1})"""
    centers: onpt.NDArray[onp.floating]
    """(N, 3)."""
    rgbs: onpt.NDArray[onp.floating]
    """(N, 3). Range [0, 1]."""
    opacities: onpt.NDArray[onp.floating]
    """(N, 1). Range [0, 1]."""
    covariances: onpt.NDArray[onp.floating]
    """(N, 3, 3)."""


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = onp.frombuffer(splat_buffer, dtype=onp.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(onp.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = onp.array([RigidTransform.rotation_from_quaternion(wxyz) for wxyz in wxyzs])
    covariances = onp.einsum(
        "nij,njk,nlk->nil", Rs, onp.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(onp.float32)
    if center:
        centers -= onp.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
        "ids": dict()
    }

def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = onp.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = onp.exp(onp.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = onp.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * onp.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + onp.exp(-v["opacity"][:, None]))
    
    #Rs = tf.SO3(wxyzs).as_matrix()
    Rs = onp.array([RigidTransform.rotation_from_quaternion(quat) for quat in wxyzs])
    covariances = onp.einsum(
        "nij,njk,nlk->nil", Rs, onp.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= onp.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
        "ids": dict()
    }

def visualize_clusters(ply_file_path: Path, center: bool = False):
    gaussians = load_ply_file(ply_file_path, center)
    cluster_path = ply_file_path.parent.parent.joinpath("clusters.npy")
    clusters = onp.load(cluster_path, allow_pickle=True)
    
    labels = clusters[0]    
    cluster_gaussians_no = []
    for group_no in range(1, len(clusters)):
        cluster_gaussians_no.extend(clusters[group_no])
        
        for gauss in clusters[group_no]:
            gaussians["ids"][gauss] = int(labels[gauss].item())
    
    new_gaussians = {key : gaussians[key][cluster_gaussians_no] for key in gaussians if key != "ids"}
    new_gaussians["ids"] = gaussians["ids"]
    return new_gaussians

def transform_gaussian(center, covariance, transformation):
    ## jyus code for gaussian updates
    # xyz0 = self.traj[0]
    
    # for i, mask in enumerate(self.group_masks_local):
    #     rigid_transform_mat = frame[i].as_matrix()
    #     rigid_transform_mat[:3,3] = rigid_transform_mat[:3,3] - xyz0[i].translation()
    #     # print(f"Object {i}: ", rigid_transform_mat)
    #     means_centered = torch.subtract(self.init_means[mask], self.init_means[mask].mean(dim=0))
    #     means_centered_homog = torch.cat([means_centered, torch.ones(means_centered.shape[0], 1).to(self.device)], dim=1)
    #     # import pdb; pdb.set_trace()
    #     self.model.gauss_params["means"][mask] = ((torch.from_numpy(rigid_transform_mat).to(torch.float32).cuda() @ means_centered_homog.T).T)[:, :3] + self.init_means[mask].mean(dim=0)
    #     self.model.gauss_params["quats"][mask] = torch.Tensor(
    #         Rot.from_matrix(
    #         torch.matmul(torch.from_numpy(frame[i].as_matrix()[:3,:3]).to(torch.float32).cuda(), quat_to_rotmat(self.init_quats[mask])).cpu()
    #         ).as_quat()).to(self.device)[:, [3, 0, 1, 2]]
    
    # Convert the center to homogeneous coordinates
    center_homogeneous = onp.append(center, 1)
    
    # Apply the transformation to the center
    transformed_center_homogeneous = transformation @ center_homogeneous
    transformed_center = transformed_center_homogeneous[:3]

    # Extract the 3x3 rotation part of the transformation matrix
    rotation_matrix = transformation[:3, :3]
    
    # Apply the rotation part to the covariance matrix
    transformed_covariance = rotation_matrix @ covariance @ rotation_matrix.T

    return transformed_center, transformed_covariance

def realtime_tracking(gaussians : SplatFile, traj_path: Path, server : vsplat.ViserServer):
    curr_gaussians = gaussians.copy()
    trajectories = onp.load(traj_path, allow_pickle=True)
    unique_clusters = sorted(set(curr_gaussians["ids"].values()))
    ind = 0
    print("starting", len(trajectories))
    gs_handles = [0 for _ in range(len(trajectories))]
    net_tf = dict()
    while ind < len(trajectories):
        ind = ind % len(trajectories)
        if ind == 0:
            curr_gaussians = gaussians
        
        # remove_button = server.gui.add_button(f"Remove splat object {ind}")

        # @remove_button.on_click
        # def _(_, gs_handle=gs_handles[ind]) -> None:
        #     gs_handle.remove()
        #     remove_button.remove()
        print("adding gs")
        
        time.sleep(10.0)
        
        transforms = trajectories[ind]
        for i, label in enumerate(unique_clusters):
            quat, pos = transforms[i].rotation(), transforms[i].translation()
            rotation = RigidTransform.rotation_from_quaternion(quat.wxyz)
            rtf = RigidTransform(rotation=rotation, translation=pos)
            
            if not label in net_tf:
                net_tf[label] = rtf.matrix
                print(f"before {label}: {rtf.matrix}")
            else:
                net_tf[label] = net_tf[label] @ rtf.matrix
    
    for label in net_tf:
        print(f"after {label}: {net_tf[label]}")
    
    import pdb; pdb.set_trace()    
        
        
        #import pdb; pdb.set_trace()
    new_gaussians_centers = []
    new_gaussians_cov = []
    for i, id in enumerate(gaussians["ids"]):
        label = int(gaussians["ids"][id])
        center = gaussians["centers"][i]
        cov = gaussians["covariances"][i]
        
        tf_center, tf_cov = transform_gaussian(center, cov, net_tf[label].matrix)
        new_gaussians_centers.append(tf_center)
        new_gaussians_cov.append(tf_cov)
        
    print("before:", curr_gaussians["centers"])
    curr_gaussians["centers"] = onp.array(new_gaussians_centers)
    curr_gaussians["covariances"] = onp.array(new_gaussians_cov)
    print("after:", curr_gaussians["centers"])
    gs_handles[ind].remove()
    ind += 1
    print("calculation done", ind)
    
    time.sleep(0.25)
    
    
    gs_handles[ind] = server.scene._add_gaussian_splats(
            f"/realtime/post_traj",
            centers=curr_gaussians["centers"],
            rgbs=curr_gaussians["rgbs"],
            opacities=curr_gaussians["opacities"],
            covariances=curr_gaussians["covariances"],
    )
    
    while True:
        time.sleep(10.0)
    return 

def main(splat_paths: tuple[Path, ...], traj_path: Annotated[Optional[Path], typer.Argument()] = None) -> None:
    server = vsplat.ViserServer()
    server.configure_theme(dark_mode=True)
    gui_reset_up = server.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: vsplat.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = RigidTransform.rotation_from_quaternion(client.camera.wxyz) @ onp.array(
            [0.0, -1.0, 0.0]
        )

    for i, splat_path in enumerate(splat_paths):
        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=True)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=True)     
            #print(splat_data["centers"])   
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")
        
        import pdb;pdb.set_trace()
        server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene._add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )
        print("added gaussian splat object", i)
        
        remove_button = server.gui.add_button(f"Remove splat object {i}")

        @remove_button.on_click
        def _(_, gs_handle=gs_handle) -> None:
            gs_handle.remove()
            remove_button.remove()
    
    if traj_path == None:
        while True:
            time.sleep(10.0)
    else:
        #gs_handle.remove()
        realtime_tracking(splat_data, traj_path, server)


if __name__ == "__main__":
    tyro.cli(main)