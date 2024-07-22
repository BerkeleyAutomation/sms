import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional
from pathlib import Path

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.math import intersect_aabb, intersect_obb
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
import trimesh
# from nerfstudio.data.scene_box import OrientedBox

# from sms.data.sms_datamanager import (
#     smsDataManager,
#     smsDataManagerConfig,
# )

# import viser
# import viser.transforms as vtf
# import trimesh
# import open3d as o3d
# import cv2
# from copy import deepcopy

from dataclasses import dataclass, field
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viewer_elements import ViewerCheckbox
from nerfstudio.models.base_model import ModelConfig
from sms.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
# from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig
from sms.model.sms_gaussian_splatting import smsGaussianSplattingModelConfig, SH2RGB
# from sms.monodepth.zoedepth_network import ZoeDepthNetworkConfig
from torch.cuda.amp.grad_scaler import GradScaler
from torchvision.transforms.functional import resize
from nerfstudio.configs.base_config import InstantiateConfig
# from lerf.utils.camera_utils import deproject_pixel, get_connected_components, calculate_overlap, non_maximum_suppression
from sms.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
# from gsplat.sh import spherical_harmonics, num_sh_bases
# from gsplat.cuda_legacy._wrapper import num_sh_bases
from sms.data.full_images_datamanager import FullImageDatamanagerConfig
from sklearn.neighbors import NearestNeighbors

from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from scipy.spatial.transform import Rotation as Rot
from typing import Literal, Type, Optional, List, Tuple, Dict
from nerfstudio.viewer.viewer_elements import *
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import math
from scipy.spatial.distance import cdist
import open3d as o3d
import tqdm


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )
def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def get_clip_patchloader(image, pipeline, image_scale):
    clip_cache_path = Path("dummy_cache2.npy")
    import time
    model_name = str(time.time())
    image = image.permute(2,0,1)[None,...]
    patchloader = PatchEmbeddingDataloader(
        cfg={
            "tile_ratio": image_scale,
            "stride_ratio": .25,
            "image_shape": list(image.shape[2:4]),
            "model_name": model_name,
        },
        device='cuda:0',
        model=pipeline.image_encoder,
        image_list=image,
        cache_path=clip_cache_path,
    )
    return patchloader

def get_grid_embeds_patch(patchloader, rn, cn, im_h, im_w, img_scale):
    "create points, which is a meshgrid of x and y coordinates, with a z coordinate of 1.0"
    r_res = im_h // rn
    c_res = im_w // cn
    points = torch.stack(torch.meshgrid(torch.arange(0, im_h,r_res), torch.arange(0,im_w,c_res)), dim=-1).cuda().long()
    points = torch.cat([torch.zeros((*points.shape[:-1],1),dtype=torch.int64,device='cuda'),points],dim=-1)
    embeds = patchloader(points.view(-1,3))
    return embeds, points

def get_2d_embeds(image: torch.Tensor, scale: float, pipeline):
    # pyramid = get_clip_pyramid(image,pipeline=pipeline,image_scale=scale)
    # embeds,points = get_grid_embeds(pyramid,image.shape[0]//resolution,image.shape[1]//resolution,image.shape[0],image.shape[1],scale)
    patchloader = get_clip_patchloader(image, pipeline=pipeline, image_scale=scale)
    embeds, points = get_grid_embeds_patch(patchloader, image.shape[0] * scale,image.shape[1] * scale, image.shape[0], image.shape[1], scale)
    return embeds, points


@dataclass
class smsdataPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: smsdataPipeline)
    """target class to instantiate"""
    datamanager: FullImageDatamanagerConfig = FullImageDatamanagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = smsGaussianSplattingModelConfig()
    """specifies the model config"""
    # depthmodel:InstantiateConfig = ZoeDepthNetworkConfig()
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""

class smsdataPipeline(VanillaPipeline):
    def __init__(
        self,
        config: smsdataPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        # highres_downscale : float = 4.0,
        use_clip : bool = True,
        # model_name : str = "dino_vits8",
        # dino_thres : float = 0.4, 
        # clip_out_queue : Optional[mp.Queue] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        # self.clip_out_queue = clip_out_queue
        # self.dino_out_queue = dino_out_queue
        self.datamanager: FullImageDatamanagerConfig = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank
        )
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        self.datamanager.to(device)
        # self.image_encoder: BaseImageEncoder = config.network.setup()
        # TODO(ethan): get rid of scene_bounds from the model

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            grad_scaler=grad_scaler,
            image_encoder=self.datamanager.image_encoder,
            datamanager=self.datamanager,
            seed_points=seed_pts,
        )
        self.model.to(device)

        # self.depthmodel = config.depthmodel.setup()

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        self.use_clip = use_clip
        self.plot_verbose = True
        
        self.img_count = 0

        self.viewer_control = self.model.viewer_control

        self.state_stack = []

        self.a_interaction_method = ViewerDropdown(
            "Interaction Method",
            default_value="Interactive",
            options=["Interactive", "Clustering"],
            cb_hook=self._update_interaction_method
        )

        self.click_gaussian = ViewerButton(name="Click", cb_hook=self._click_gaussian)
        self.click_location = None
        self.click_handle = None


        self.crop_to_click = ViewerButton(name="Crop to Click", cb_hook=self._crop_to_click, disabled=True)
        # self.crop_to_group_level = ViewerSlider(name="Group Level", min_value=0, max_value=29, step=1, default_value=15, cb_hook=self._update_crop_vis, disabled=False)
        self.crop_group = []

        self.add_crop_to_group_list = ViewerButton(name="Add Crop to Group List", cb_hook=self._add_crop_to_group_list, disabled=True)
        self.add_crop_to_previous_group = ViewerButton(name="Add Crop to Previous Group", cb_hook=self._add_crop_to_previous_group, disabled=True)
        self.view_crop_group_list = ViewerButton(name="View Crop Group List", cb_hook=self._view_crop_group_list, disabled=True)
        self.crop_group_list = []
        self.keep_inds = None

        self.move_current_crop = ViewerButton(name="Drag Current Crop", cb_hook=self._drag_current_crop, disabled=True)
        self.crop_transform_handle = None

        self.reset_state = ViewerButton(name="Reset State", cb_hook=self._reset_state, disabled=True)

        self.z_export_options = ViewerCheckbox(name="Export Options", default_value=False, cb_hook=self._update_export_options)
        self.z_export_options_visible_gaussians = ViewerButton(
            name="Export Visible Gaussians",
            visible=False,
            cb_hook=self._export_visible_gaussians
            )
        self.z_export_options_cluster_labels = ViewerButton(
            name="Export Cluster",
            visible=False,
            cb_hook=self._export_clusters
            )


    # this only calcualtes the features for the given image
    def add_image(
        self,
        img: torch.Tensor, 
        pose: Cameras = None, 
    ):

        self.datamanager.add_image(img)
        # self.img_count += 1

    # this actually adds the image to the datamanager + dataset...?
    # @profile
    def process_image(
        self,
        img: torch.Tensor, 
        depth: torch.Tensor,
        pose: Cameras, 
        clip: dict,
        dino,
        downscale_factor = 1,
    ):
        print("Adding image to train dataset",pose.camera_to_worlds[:3,3].flatten())
        
        self.datamanager.process_image(img, depth, pose, clip, dino, downscale_factor)
        self.img_count += 1

    def add_to_clip(self, clip: dict, step: int):
        self.datamanager.add_to_clip(clip, step)

    def _queue_state(self):
        """Save current state to stack"""
        import copy
        self.state_stack.append(copy.deepcopy({k:v.detach() for k,v in self.model.gauss_params.items()}))
        self.reset_state.set_disabled(False)


    def _reset_state(self, button: ViewerButton, pop = True):
        """Revert to previous saved state"""

        assert len(self.state_stack) > 0, "No previous state to revert to"
        if pop:
            prev_state = self.state_stack.pop()
        else:
            prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name]

        self.click_location = None
        if self.click_handle is not None:
            self.click_handle.remove()
        self.click_handle = None

        self.click_gaussian.set_disabled(False)

        self.crop_to_click.set_disabled(True)
        # self.crop_to_group_level.set_disabled(True)
        # self.crop_to_group_level.value = 0.5
        self.move_current_crop.set_disabled(True)
        self.crop_group = []
        if self.crop_transform_handle is not None:
            self.crop_transform_handle.remove()
            self.crop_transform_handle = None
        if len(self.state_stack) == 0:
            self.reset_state.set_disabled(True)

        # self.cluster_labels = None
        # self.model.cluster_scene.set_disabled(False)
        self.add_crop_to_group_list.set_disabled(True)

    def _reset_crop_group_list(self, button: ViewerButton):
        """Reset the crop group list"""
        self.crop_group_list = []
        self.keep_inds = []
        self.add_crop_to_group_list.set_disabled(True)
        self.view_crop_group_list.set_disabled(True)
    
    def _add_crop_to_group_list(self, button: ViewerButton):
        """Add the current crop to the group list"""
        self.crop_group_list.append(self.crop_group[0])
        self._reset_state(None, pop=False)
        self.view_crop_group_list.set_disabled(False)
    
    def _add_crop_to_previous_group(self, button: ViewerButton):
        """Combine the current crop with the previous group"""
        self.crop_group_list[-1] = torch.cat([self.crop_group_list[-1], self.crop_group[0]])
        self._reset_state(None, pop=False)
        self.view_crop_group_list.set_disabled(False)

    def _view_crop_group_list(self, button: ViewerButton):
        if len(self.crop_group_list) == 0:
            return
        if len(self.state_stack) == 0:
            return

        keep_inds = []
        for inds in self.crop_group_list:
            keep_inds.extend(inds)
        keep_inds = torch.stack(keep_inds)
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name][keep_inds]
        self.keep_inds = keep_inds
        self._export_clusters(None)
        self.z_export_options_cluster_labels.visible = True

    def _crop_to_click(self, button: ViewerButton):
        """Crop to click location"""
        assert self.click_location is not None, "Need to specify click location"

        self._queue_state()  # Save current state
        curr_means = self.model.gauss_params['means'].detach()
        self.model.eval()

        # The only way to reset is to reset the state using the reset button.
        self.click_gaussian.set_disabled(True)  # Disable user from changing click
        self.crop_to_click.set_disabled(True)  # Disable user from changing click

        # Get the 3D location of the click
        location = self.click_location
        location = torch.tensor(location).view(1, 3).to(self.device)

        # The list of positions to query for garfield features. The first one is the click location.
        positions = torch.cat([location, curr_means])  # N x 3

        # Create a kdtree, to get the closest gaussian to the click-point.
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curr_means.cpu().numpy()))
        kdtree = o3d.geometry.KDTreeFlann(points)
        _, inds, _ = kdtree.search_knn_vector_3d(location.view(3, -1).float().detach().cpu().numpy(), 10)

        # get the closest point to the sphere, using kdtree
        sphere_inds = inds
        # scales = torch.ones((positions.shape[0], 1)).to(self.device)

        keep_list = []

        if self.model.cluster_labels == None:
            instances = self.model.get_grouping_at_points(positions)  # (1+N, 256)
            click_instance = instances[0]
            # import pdb; pdb.set_trace()
            affinity = torch.norm(click_instance - instances, dim=1)[1:]

            # Filter out points that have affinity < 0.5 (i.e., not likely to be in the same group)
            keeps = torch.where(affinity > 0.5)[0].cpu()
            keep_points = points.select_by_index(keeps.tolist())  # indices of gaussians

            # Here, we desire the gaussian groups to be grouped tightly together spatially. 
            # We use DBSCAN to group the gaussians together, and choose the cluster that contains the click point.
            # Note that there may be spuriously high affinity between points that are spatially far apart,
            #  possibly due two different groups being considered together at an odd angle / far viewpoint.

            # If there are too many points, we downsample them first before DBSCAN.
            # Then, we assign the filtered points to the cluster of the nearest downsampled point.
            if len(keeps) > 5000:
                curr_point_min = keep_points.get_min_bound()
                curr_point_max = keep_points.get_max_bound()

                # downsample_size = 0.01 * s
                _, _, curr_points_ds_ids = keep_points.voxel_down_sample_and_trace(
                    voxel_size=0.0001,
                    min_bound=curr_point_min,
                    max_bound=curr_point_max,
                )
                curr_points_ds_ids = np.array([points[0] for points in curr_points_ds_ids])
                curr_points_ds = keep_points.select_by_index(curr_points_ds_ids)
                curr_points_ds_selected = np.zeros(len(keep_points.points), dtype=bool)
                curr_points_ds_selected[curr_points_ds_ids] = True

                _clusters = np.asarray(curr_points_ds.cluster_dbscan(eps=0.02, min_points=5))
                nn_model = NearestNeighbors(
                    n_neighbors=1, algorithm="auto", metric="euclidean"
                ).fit(np.asarray(curr_points_ds.points))

                _, indices = nn_model.kneighbors(np.asarray(keep_points.points)[~curr_points_ds_selected])

                clusters = np.zeros(len(keep_points.points), dtype=int)
                clusters[curr_points_ds_selected] = _clusters
                clusters[~curr_points_ds_selected] = _clusters[indices[:, 0]]

            else:
                clusters = np.asarray(keep_points.cluster_dbscan(eps=0.02, min_points=5))

            # Choose the cluster that contains the click point. If there is none, move to the next scale.
            cluster_inds = clusters[np.isin(keeps, sphere_inds)]
            cluster_inds = cluster_inds[cluster_inds != -1]

            cluster_ind = cluster_inds[0]

            keeps = keeps[np.where(clusters == cluster_ind)]


            keep_list.append(keeps)

            if len(keep_list) == 0:
                print("No gaussians within crop, aborting")
                # The only way to reset is to reset the state using the reset button.
                self.click_gaussian.set_disabled(False)
                self.crop_to_click.set_disabled(False)
                return
        else:
            # Handle case where we have precomputed cluster labels
            vote = int(torch.tensor(self.model.cluster_labels[sphere_inds].mode())[0].item()) # mode group ID from the click sphere samples
            
            keep_inds_list = torch.where(self.model.cluster_labels == vote)[0] # get all points in the same group as the click sphere samples
            keep_points_o3d = points.select_by_index(keep_inds_list.tolist()) # clustered points in o3d format for DBSCAN
            
            sphere_ind_vote = torch.where(self.model.cluster_labels[sphere_inds] == vote)[0]
            
            sphere_inds_keep = [(torch.where(keep_inds_list == torch.tensor(sphere_inds)[i])[0]).item() for i in sphere_ind_vote.tolist()]
            # Secondary clustering in cartesian space to filter outliers
            group_clusters = keep_points_o3d.cluster_dbscan(eps=0.013, min_points=1)
            inner_vote = torch.tensor(group_clusters)[sphere_inds_keep].mode()[0].item()
            keep_inds_list_inner = torch.where(torch.tensor(group_clusters) == inner_vote)[0]
            keep_list = [keep_inds_list[keep_inds_list_inner]]
        
        # Remove the click handle + visualization
        self.click_location = None
        self.click_handle.remove()
        self.click_handle = None
        
        self.crop_group = keep_list
        
        self.add_crop_to_group_list.set_disabled(False)
        if len(self.crop_group_list) > 0:
            self.add_crop_to_previous_group.set_disabled(False)
        self.move_current_crop.set_disabled(False)

        keep_inds = self.crop_group[0]
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name][keep_inds]

    def _drag_current_crop(self, button: ViewerButton):
        """Add a transform control to the current scene, and update the model accordingly."""
        self.crop_to_group_level.set_disabled(True)  # Disable user from changing crop
        self.move_current_crop.set_disabled(True)  # Disable user from creating another drag handle
        
        scene_centroid = self.model.gauss_params['means'].detach().mean(dim=0)
        self.crop_transform_handle = self.viewer_control.viser_server.add_transform_controls(
            name=f"/scene_transform",
            position=(VISER_NERFSTUDIO_SCALE_RATIO*scene_centroid).cpu().numpy(),
        )

        # Visualize the whole scene -- the points corresponding to the crop will be controlled by the transform handle.
        crop_inds = self.crop_group[self.crop_to_group_level.value]
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name].clone()

        curr_means = self.model.gauss_params['means'].clone().detach()
        curr_rotmats = quat_to_rotmat(self.model.gauss_params['quats'][crop_inds].detach())

        @self.crop_transform_handle.on_update
        def _(_):
            handle_position = torch.tensor(self.crop_transform_handle.position).to(self.device)
            handle_position = handle_position / VISER_NERFSTUDIO_SCALE_RATIO
            handle_rotmat = quat_to_rotmat(torch.tensor(self.crop_transform_handle.wxyz).to(self.device).float())

            means = self.model.gauss_params['means'].detach()
            quats = self.model.gauss_params['quats'].detach()

            means[crop_inds] = handle_position.float() + torch.matmul(
                handle_rotmat, (curr_means[crop_inds] - curr_means[crop_inds].mean(dim=0)).T
            ).T
            quats[crop_inds] = torch.Tensor(Rot.from_matrix(
                torch.matmul(handle_rotmat.float(), curr_rotmats.float()).cpu().numpy()
            ).as_quat()).to(self.device)  # this is in xyzw format
            quats[crop_inds] = quats[crop_inds][:, [3, 0, 1, 2]]  # convert to wxyz format

            self.model.gauss_params['means'] = torch.nn.Parameter(means.float())
            self.model.gauss_params['quats'] = torch.nn.Parameter(quats.float())

            self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender

    def _update_interaction_method(self, dropdown: ViewerDropdown):
        """Update the UI based on the interaction method"""
        hide_in_interactive = (not (dropdown.value == "Interactive")) # i.e., hide if in interactive mode

        self.cluster_scene.set_hidden((not hide_in_interactive))
        self.cluster_scene_scale.set_hidden((not hide_in_interactive))
        self.cluster_scene_shuffle_colors.set_hidden((not hide_in_interactive))

        self.click_gaussian.set_hidden(hide_in_interactive)
        self.crop_to_click.set_hidden(hide_in_interactive)
        self.crop_to_group_level.set_hidden(hide_in_interactive)
        self.move_current_crop.set_hidden(hide_in_interactive)

    def _click_gaussian(self, button: ViewerButton):
        """Start listening for click-based 3D point specification.
        Refer to garfield_interaction.py for more details."""
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.click_gaussian.set_disabled(False)
            self.crop_to_click.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Refer to garfield_interaction.py for more details."""

        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf

        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        # project it into coordinates with matrix
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.model.eval()
        outputs = self.model.get_outputs(cam.to(self.device))
        self.model.train()
        with torch.no_grad():
            # if (pix_y > outputs["depth"].shape[0] or 
            #     pix_x > outputs["depth"].shape[1] or
            #     pix_x < 0 or 
            #     pix_y < 0):
            #     print("Click bug triggered")
            #     return False
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()

        self.click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)

        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # type: ignore
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/click",
            mesh=sphere_mesh,
            position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location,
        )
        
    def _update_export_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        self.z_export_options_visible_gaussians.set_hidden(not checkbox.value)
    
    def _export_clusters(self, button: ViewerButton):
        """Export the cluster information to a .npy file"""
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        filename = Path(output_dir) / f"clusters.npy"
        
        if self.model.cluster_labels is not None and self.keep_inds is not None:
            np.save(filename, np.array([self.model.cluster_labels, self.keep_inds], dtype=object))
        else:
            print("No cluster labels to export")
            
    def _export_visible_gaussians(self, button: ViewerButton):
        """Export the visible gaussians to a .ply file"""
        # location to save
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        filename = Path(output_dir) / f"gaussians.ply"

        # Copied from exporter.py
        from collections import OrderedDict
        map_to_tensors = OrderedDict()
        model=self.model

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]
                    
                assert shs_0.shape[1] == 3
                map_to_tensors[f"red"] = np.array(SH2RGB(shs_0[:, 0])*255).astype(np.uint8)
                map_to_tensors[f"green"] = np.array(SH2RGB(shs_0[:, 1])*255).astype(np.uint8)
                map_to_tensors[f"blue"] = np.array(SH2RGB(shs_0[:, 2])*255).astype(np.uint8)

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)
        from nerfstudio.scripts.exporter import ExportGaussianSplat
        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)

    # def _export_visible_gaussians(self, button: ViewerButton):
    #     """Export the visible gaussians to a .ply file"""
    #     output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
    #     filename = Path(output_dir) / f"prime_gaussians.ply"

    #     # Copied from exporter.py
    #     from collections import OrderedDict
    #     map_to_tensors = OrderedDict()
    #     model=self.model

    #     with torch.no_grad():
    #         positions = model.means.cpu().numpy()
    #         if model.config.sh_degree > 0:
    #             shs_0 = model.shs_0.contiguous().cpu().numpy()
    #         colors = model.colors.cpu().numpy()
    #         normalized_colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    #         # import viser
    #         # new_colors_viser = viser.ViserServer()
    #         positions = positions.astype('float64')
    #         normalized_colors = normalized_colors.astype('float64')
    #         # new_colors_viser.add_point_cloud('ply_pc',positions,normalized_colors,point_size=0.001)
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(positions)
    #         pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
    #         o3d.io.write_point_cloud(str(filename),pcd)
        