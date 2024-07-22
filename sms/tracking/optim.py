import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import trimesh
from typing import Optional, List, Tuple
import moviepy.editor as mpy
from copy import deepcopy

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from threading import Lock
import warp as wp
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
# from garfield.garfield_gaussian_pipeline import GarfieldGaussianPipeline
from sms.sms_data_pipeline import smsdataPipeline
from nerfstudio.utils import writer
from nerfstudio.models.splatfacto import SH2RGB

from sms.tracking.rigid_group_optimizer import RigidGroupOptimizer
from sms.tracking.toad_object import ToadObject
from sms.tracking.frame import Frame
from sms.data.utils.dino_dataloader2 import DinoDataloader
import os.path as osp
from sms.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork


class Optimizer:
    """Wrapper around 1) RigidGroupOptimizer and 2) GraspableToadObject.
    Operates in camera frame, not world frame."""

    init_cam_pose: torch.Tensor
    """Initial camera pose, in OpenCV format.
    This is aligned with the camera pose provided in __init__,
    and is in world coordinates/scale."""
    viewer_ns: Viewer
    """Viewer for nerfstudio visualization (not the same as robot visualization)."""

    num_groups: int
    """Number of object parts."""

    toad_object: ToadObject
    """Meshes + grasps for object parts."""

    optimizer: RigidGroupOptimizer
    """Optimizer for part poses."""
    MATCH_RESOLUTION: int = 500
    """Camera resolution for RigidGroupOptimizer."""

    initialized: bool = False
    """Whether the object pose has been initialized. This is set to `False` at `ToadOptimizer` initialization."""

    def __init__(
        self,
        config_path: Path,  # path to the nerfstudio config file
        ply: Path,  # path to the object ply file
        K: np.ndarray,  # camera intrinsics
        width: int,  # camera width
        height: int,  # camera height
        init_cam_pose: torch.Tensor,  # initial camera pose in OpenCV format
    ):
        if ply.exists():
            self.ply = ply
        else:
            print(f"PLY file {ply} does not exist. Proceed with interactive clustering.")
        
        # Load the smsdataPipeline.
        train_config, self.pipeline, _, _ = eval_setup(config_path)
        assert isinstance(self.pipeline, smsdataPipeline)
        train_config.logging.local_writer.enable = False

        assert self.pipeline.datamanager.train_dataset is not None
        dataset_scale = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale

        writer.setup_local_writer(train_config.logging, max_iter=train_config.max_num_iterations)
        self.viewer_ns = Viewer(
            ViewerConfig(
                default_composite_depth=False,
                num_rays_per_chunk=-1
            ),
            config_path.parent,
            self.pipeline.datamanager.get_datapath(),
            self.pipeline,
            train_lock=Lock()
        )
        assert self.viewer_ns.train_lock is not None

        self.keep_inds = None

        self.group_labels, self.group_masks, self.group_masks_global = self._setup_crops_and_groups()
        self.max_relevancy_label = None
        self.max_relevancy_text = None

        self.num_groups = len(self.group_masks)
        
        # Init DINO dataloader for 'Frames' extractor_fn
        cache_dir = config_path.parent.parent.parent
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        
        self.dino_dataloader = DinoDataloader(
            image_list = None,
            device = 'cuda',
            cfg={"image_shape": [719, 1279]}, #HARDCODED BAD
            cache_path=dino_cache_path,
            use_denoiser=False,
        )
        
        assert init_cam_pose.shape == (1, 3, 4)
        self.init_cam_pose = deepcopy(init_cam_pose)

        # For nerfstudio, feed the camera as:
        #  - opengl format
        #  - in nerfstudio scale
        #  - as `Cameras` object
        #  - with `MATCH_RESOLUTION` resolution.
        init_cam_ns = torch.cat([
            init_cam_pose[0], torch.tensor([0, 0, 0, 1], dtype=torch.float32).reshape(1, 4)
        ], dim=0) @ (torch.from_numpy(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])).float())
        init_cam_ns = init_cam_ns[None, :3, :]
        init_cam_ns[:, :3, 3] = init_cam_ns[:, :3, 3] * dataset_scale  # convert to meters
        assert init_cam_ns.shape == (1, 3, 4)

        cam2world_ns = Cameras(
            camera_to_worlds=init_cam_ns,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=width,
            height=height,
        )
        cam2world_ns.rescale_output_resolution(
            self.MATCH_RESOLUTION / max(width, height)
        )
        
        # Set up the optimizer.
        self.cam2world_ns = cam2world_ns
        self.dataset_scale = dataset_scale

        self.orig_means = self.pipeline.model.gauss_params["means"].detach().clone()
        self.orig_quats = self.pipeline.model.gauss_params["quats"].detach().clone()

        self.optimizer = RigidGroupOptimizer(
            self.pipeline.model,
            group_masks=self.group_masks,
            group_labels=self.group_labels,
            dataset_scale=dataset_scale,
            render_lock=self.viewer_ns.train_lock,
        )

        # Initialize the object -- remember that ToadObject works in world scale,
        # since grasps + etc are in world scale.
        start = time.time()
        self.toad_object = ToadObject.from_points_and_clusters(
            self.optimizer.init_means.detach().cpu().numpy(),
            self.optimizer.group_labels.detach().cpu().numpy(),
            scene_scale=self.optimizer.dataset_scale,
        )
        print(f"Time taken for init (object): {time.time() - start:.2f} s")

        self.initialized = False

    def reset_optimizer(self) -> None:
        """Re-generate self.optimizer."""
        self.optimizer.reset_transforms()
        del self.optimizer
        self.pipeline.model.gauss_params["means"] = self.orig_means.clone()
        self.pipeline.model.gauss_params["quats"] = self.orig_quats.clone()
        self.optimizer = RigidGroupOptimizer(
            self.pipeline.model,
            # self.pipeline.datamanager.dino_dataloader,
            self.cam2world_ns,
            group_masks=self.group_masks,
            group_labels=self.group_labels,
            dataset_scale=self.dataset_scale,
            render_lock=self.viewer_ns.train_lock,
        )

    def _setup_crops_and_groups(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Set up the crops and groups for the optimizer, interactively."""
        cluster_labels = None
        try:
            # self.pipeline.load_state()
            if getattr(self.pipeline.model, "best_scales") is None:
                raise TypeError
            _ = input("Model populated (interactively crop and press enter to continue)")
            self.keep_inds = self.pipeline.keep_inds
            cluster_labels = self.pipeline.model.cluster_labels[self.keep_inds].to(torch.int32)
        except TypeError:
            print("Model not populated yet. Please wait...")
            # Wait for the user to set up the crops and groups.
            while getattr(self.pipeline.model, "best_scales") is None:
                time.sleep(0.1)
            # while True:
            #     time.sleep(0.1)
            _ = input("Model populated (interactively crop and press enter to continue)")
            self.keep_inds = self.pipeline.keep_inds
            
            keep_inds_mask = torch.zeros_like(self.pipeline.model.cluster_labels)
            keep_inds_mask[self.keep_inds] = 1
            keep_inds_mask = keep_inds_mask.to(torch.bool)
            
            cluster_labels = self.pipeline.model.cluster_labels[self.keep_inds].to(torch.int32)
            cluster_labels_global = self.pipeline.model.cluster_labels.to(torch.int32)

        mapping, cluster_labels_keep = torch.unique(cluster_labels, return_inverse=True)
        group_masks = [(cid == cluster_labels_keep).cuda() for cid in range(cluster_labels_keep.max().item() + 1)]
        
        group_masks_global = [((cid == cluster_labels_global) & keep_inds_mask).cuda() for cid in mapping]
        
        return cluster_labels_keep.int().cuda(), group_masks, group_masks_global

    def set_frame(self, rgb, ns_camera, depth) -> None:
        """Set the frame for the optimizer -- doesn't optimize the poses yet."""
        target_frame_rgb = (rgb/255)
        
        frame = Frame(rgb=target_frame_rgb, camera=ns_camera, dino_fn=self.dino_dataloader.get_pca_feats, metric_depth_img=depth)
        
        self.optimizer.set_frame(frame)

    def init_obj_pose(self):
        """Initialize the object pose, and render the object pose optimization process.
        Also updates `initialized` to `True`."""
        # retval only matters for visualization
        start = time.time()
        renders = self.optimizer.initialize_obj_pose(render=True,n_seeds=7)
        print(f"Time taken for init (pose opt): {time.time() - start:.2f} s")

        start = time.time()
        if len(renders)>1:
            renders = [r.detach().cpu().numpy()*255 for r in renders]
            # save video as test_camopt.mp4
            out_clip = mpy.ImageSequenceClip(renders, fps=30)  
            out_clip.write_videofile("test_camopt.mp4")
        print(f"Time taken for init (video): {time.time() - start:.2f} s")
        
        # Assert there are no nans in part_deltas
        assert not torch.isnan(self.optimizer.part_deltas).any().item()
        if torch.isnan(self.optimizer.part_deltas).any().item():
            exit()
        self.initialized = True

    def step_opt(self,niter):
        """Run the optimizer for `niter` iterations."""
        assert self.initialized, "Please initialize the object pose first."
        outputs = self.optimizer.step(niter=niter)
        return outputs

    def get_pointcloud(self) -> trimesh.PointCloud:
        """Get the pointcloud of the object parts in camera frame."""
        # c2w = self.cam2world.camera_to_worlds.squeeze()  # (3, 4)
        # parts2cam = self.optimizer.get_poses_relative_to_camera(c2w)  # --> (N, 4, 4)
        with torch.no_grad():
            self.optimizer.apply_to_model(self.optimizer.part_deltas, self.optimizer.centroids, self.optimizer.group_labels)
        points = self.optimizer.dig_model.means.clone().detach()
        colors = SH2RGB(self.optimizer.dig_model.colors.clone().detach())
        points = points / self.optimizer.dataset_scale
        pc = trimesh.PointCloud(points.cpu().numpy(), colors=colors.cpu().numpy())  # pointcloud in world frame

        cam2world = torch.cat([
            self.init_cam_pose.squeeze(),
            torch.Tensor([[0, 0, 0, 1]]).to(self.init_cam_pose.device)
        ], dim=0)
        pc.vertices = trimesh.transform_points(
            pc.vertices,
            cam2world.inverse().cpu().numpy()
        )  # pointcloud in camera frame.
        return pc

    def get_parts2cam(self,keyframe=None) -> List[vtf.SE3]:
        """Get the parts' poses in camera frame. Wrapper for `RigidGroupOptimizer.get_poses_relative_to_camera`."""
        # Note: `get_poses_relative_to_camera` has dataset_scale scaling built in.
        parts2cam = self.optimizer.get_poses_relative_to_camera(self.init_cam_pose.squeeze().cuda(),keyframe=keyframe)

        # Convert to vtf.SE3.
        parts2cam_vtf = [
            vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_matrix(pose[:3,:3].cpu().numpy()),
                translation=pose[:3,3].cpu().numpy()
            ) for pose in parts2cam
        ]
        return parts2cam_vtf
    
    def get_parts2world(self,keyframe=None) -> List[vtf.SE3]:
        """Get the parts' poses in world frame. Wrapper for `RigidGroupOptimizer.get_poses_relative_to_camera`."""
        # Note: `get_poses_relative_to_camera` has dataset_scale scaling built in.
        parts2world = self.optimizer.get_part_poses(keyframe=keyframe)

        # Convert to vtf.SE3.
        parts2world_vtf = [
            vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_matrix(pose[:3,:3].cpu().numpy()),
                translation=pose[:3,3].cpu().numpy()
            ) for pose in parts2world
        ]
        return parts2world_vtf

    def get_hands(self, keyframe: int) -> List[trimesh.Trimesh]:
        """Get hands in camera frame."""
        self.optimizer.apply_keyframe(keyframe)
        hands = [deepcopy(_) for _ in self.optimizer.hand_frames[keyframe]]
        T_obj_world = self.optimizer.get_registered_o2w().cpu().numpy()
        T_world_cam = np.linalg.inv(np.concatenate([
            self.init_cam_pose.squeeze().cpu().numpy(),
            np.array([[0, 0, 0, 1]])
        ], axis=0))
        T_obj_cam = T_world_cam @ T_obj_world

        # hands are in object frame! We want it in camera frame.
        for h in hands:
            h.apply_transform(T_obj_cam)
            h.vertices = h.vertices / self.optimizer.dataset_scale

        return hands
    
    def get_clip_relevancy(self, clip_encoder: OpenCLIPNetwork) -> int:

        init_means = self.optimizer.init_means # (N, 3)
        distances, indicies = self.optimizer.sms_model.k_nearest_sklearn(init_means, 3, True)
        distances = torch.from_numpy(distances).to(self.optimizer.sms_model.device)
        indicies = torch.from_numpy(indicies).view(-1)
        weights = torch.sigmoid(self.optimizer.init_opacities[indicies].view(-1, 4))
        weights = torch.nn.Softmax(dim=-1)(weights)
        points = init_means[indicies]
        
        hash_encoding = self.optimizer.sms_model.gaussian_field.get_hash(points) # (N, 3) -> (N, 96)
        hash_encoding = hash_encoding.view(-1, 4, hash_encoding.shape[1])
        hash_encoding = (hash_encoding * weights.unsqueeze(-1))
        hash_encoding = hash_encoding.sum(dim=1)
        
        clip_feats = self.optimizer.sms_model.gaussian_field.get_clip_outputs_from_feature(hash_encoding, 
            self.optimizer.sms_model.best_scales[0].to(self.optimizer.sms_model.device) * 
            torch.ones(self.optimizer.sms_model.num_points, 1, device=self.optimizer.sms_model.device)) # (N, 96) -> (N, 512)
        
        relevancy = clip_encoder.get_relevancy(clip_feats / (clip_feats.norm(dim=-1, keepdim=True)+1e-6), 0).view(self.optimizer.sms_model.num_points, -1)
        return relevancy
    
