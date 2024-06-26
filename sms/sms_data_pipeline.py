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
from sms.model.sms_gaussian_splatting import smsGaussianSplattingModelConfig
# from sms.monodepth.zoedepth_network import ZoeDepthNetworkConfig
from torch.cuda.amp.grad_scaler import GradScaler
from torchvision.transforms.functional import resize
from nerfstudio.configs.base_config import InstantiateConfig
# from lerf.utils.camera_utils import deproject_pixel, get_connected_components, calculate_overlap, non_maximum_suppression
from sms.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
# from gsplat.sh import spherical_harmonics, num_sh_bases
# from gsplat.cuda_legacy._wrapper import num_sh_bases
from sms.data.scene_box import SceneBox, OrientedBox
from sms.data.full_images_datamanager import FullImageDatamanagerConfig

# import sms.query_diff_utils as query_diff_utils
from sms.sms_utils import Utils as U
from typing import Literal, Type, Optional, List, Tuple, Dict
# import lerf.utils.query_diff_utils as query_diff_utils
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import math
from scipy.spatial.distance import cdist
import open3d as o3d

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

    def monodepth_inference(self, image):
        # Down-sample
        down_height = image.shape[0] // 2
        down_width = image.shape[1] // 2
        imagedown = cv2.resize(np.array(image), (down_width, down_height), interpolation=cv2.INTER_AREA)
        
        depth = self.depthmodel.get_depth(imagedown)

        depth = F.interpolate(depth, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)

        return depth
