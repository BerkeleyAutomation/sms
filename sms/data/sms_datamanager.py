# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin
import random
import cv2
import numpy as np
import torch
import time
from copy import deepcopy, copy
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode
from tqdm import tqdm

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE

from rich.progress import Console
from sms.data.sms_dataloader import smsDataloader

CONSOLE = Console(width=120)

# from lerf.data.utils.dino_dataloader import DinoDataloader
from sms.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from functools import cached_property
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
import torch.multiprocessing as mp
from sms.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
from sms.data.sms_dataparser import smsDataParserConfig
from sms.data.sms_dataset import smsDataset
from sms.data.utils.detic_dataloader import DeticDataloader
from nerfstudio.utils.misc import get_orig_class



@dataclass
class smsDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: smsDataManager)
    dataparser: AnnotatedDataParserUnion = smsDataParserConfig()
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    cache_images: Literal["no-cache", "cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "numpy", caches as numpy arrays, if "torch", caches as torch tensors."""
    patch_tile_size_range: Tuple[int, int] = (0.10, 0.45)
    """The range of tile sizes to sample from for patch-based training"""
    patch_tile_size_res: int = 7
    """The number of tile sizes to sample from for patch-based training"""
    patch_stride_scaler: float = 0.5
    """The stride scaler for patch-based training"""
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""
    clip_downscale_factor: int = 4
    """The downscale factor for the clip pyramid"""


class smsDataManager(DataManager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: smsDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
        self,
        config: smsDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        clip_out_queue: Optional[mp.Queue] = None,
        network: BaseImageEncoderConfig = BaseImageEncoderConfig(),
        **kwargs,
    ):
        self.config = config
        self.network = network
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            # CONSOLE.print("Train dataset has over 500 images, overriding cach_images to cpu", style="bold yellow")
            self.config.cache_images = "cpu"
        self.cached_train, self.cached_eval = self.cache_images(self.config.cache_images)
        # self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        # if self.config.masks_on_gpu is True:
        #     self.exclude_batch_keys_from_device.remove("mask")
        # if self.config.images_on_gpu is True:
        #     self.exclude_batch_keys_from_device.remove("image")

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        # assert len(self.train_unseen_cameras) > 0, "No data found in dataset"

        super().__init__()
        self.use_clip = False
        self.lerf_step = 0
        self.clip_out_queue = clip_out_queue


        # self.network: BaseImageEncoder = self.config.network

        # images = [self.cached_train[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.cached_train))]
        # images = torch.cat(images)
        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.network.name}"))
        # dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality
        # self.dino_dataloader = DinoDataloader(
        #     image_list=images,
        #     device=self.device,
        #     cfg={"image_shape": list(images.shape[2:4])},
        #     cache_path=dino_cache_path,
        # )
        torch.cuda.empty_cache()

        h = self.dataparser_config.img_height
        w = self.dataparser_config.img_width
        print("Input image shape: ", h, w)
        print("RGB Downresolution shape: ", self.train_dataparser_outputs.metadata['image_height'], self.train_dataparser_outputs.metadata['image_width'])

        self.clip_interpolator = PyramidEmbeddingDataloader(
            image_list=[],
            device='cuda:0',
            # device='cuda:0',
            cfg={
                "tile_size_range": list(self.config.patch_tile_size_range),
                "tile_size_res": self.config.patch_tile_size_res,
                "stride_scaler": self.config.patch_stride_scaler,
                "image_shape": [h,w],
            },
            cache_path=clip_cache_path,
            out_queue= self.clip_out_queue,
            # network=self.network,
        )
        self.clip_interpolator.start()
        # self.clip_interpolator.device = 'cuda:0' #??
        self.clip_interpolator.device = 'cuda:0'
        self.clip_interpolator.create(None, self.network.setup())

        self.detic = DeticDataloader()
        self.detic.create()
        self.detic.default_vocab()

        self.curr_scale = None
        self.random_pixels = None

    def cache_images(self, cache_images_option):
        cached_train = []
        # CONSOLE.log("Caching / undistorting train images")
        for i in tqdm(range(len(self.train_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.train_dataset.get_data(i)
            camera = self.train_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
                distortion_params = np.array(
                    [
                        distortion_params[0],
                        distortion_params[1],
                        distortion_params[4],
                        distortion_params[5],
                        distortion_params[2],
                        distortion_params[3],
                        0,
                        0,
                    ]
                )
                newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
                image = cv2.undistort(image, K, distortion_params, None, newK)
                # crop the image and update the intrinsics accordingly
                x, y, w, h = roi
                image = image[y : y + h, x : x + w]
                if "mask" in data:
                    data["mask"] = data["mask"][y : y + h, x : x + w]
                if "depth_image" in data:
                    data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
                K = newK
                # update the width, height
                self.train_dataset.cameras.width[i] = w
                self.train_dataset.cameras.height[i] = h

            elif camera.camera_type.item() == CameraType.FISHEYE.value:
                distortion_params = np.array(
                    [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
                )
                newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
                )
                # and then remap:
                image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                K = newK
            else:
                raise NotImplementedError("Only perspective and fisheye cameras are supported")
            data["image"] = torch.from_numpy(image)

            if "mask" in data:
                mask = data["mask"].numpy()
                if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
                    mask = cv2.undistort(mask, K, distortion_params, None, None)
                elif camera.camera_type.item() == CameraType.FISHEYE.value:
                    mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, None)
                else:
                    raise NotImplementedError("Only perspective and fisheye cameras are supported")
                data["mask"] = torch.from_numpy(mask)

            cached_train.append(data)

            self.train_dataset.cameras.fx[i] = float(K[0, 0])
            self.train_dataset.cameras.fy[i] = float(K[1, 1])
            self.train_dataset.cameras.cx[i] = float(K[0, 2])
            self.train_dataset.cameras.cy[i] = float(K[1, 2])

        cached_eval = []
        # CONSOLE.log("Caching / undistorting eval images")
        for i in tqdm(range(len(self.eval_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.eval_dataset.get_data(i)
            camera = self.eval_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
                distortion_params = np.array(
                    [
                        distortion_params[0],
                        distortion_params[1],
                        distortion_params[4],
                        distortion_params[5],
                        distortion_params[2],
                        distortion_params[3],
                        0,
                        0,
                    ]
                )
                newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
                image = cv2.undistort(image, K, distortion_params, None, newK)
                # crop the image and update the intrinsics accordingly
                x, y, w, h = roi
                image = image[y : y + h, x : x + w]
                if "mask" in data:
                    data["mask"] = data["mask"][y : y + h, x : x + w]
                if "depth_image" in data:
                    data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
                K = newK
                # update the width, height
                self.eval_dataset.cameras.width[i] = w
                self.eval_dataset.cameras.height[i] = h

            elif camera.camera_type.item() == CameraType.FISHEYE.value:
                distortion_params = np.array(
                    [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
                )
                newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
                )
                # and then remap:
                image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                K = newK
            else:
                raise NotImplementedError("Only perspective and fisheye cameras are supported")
            data["image"] = torch.from_numpy(image)

            if "mask" in data:
                mask = data["mask"].numpy()
                if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
                    mask = cv2.undistort(mask, K, distortion_params, None, None)
                elif camera.camera_type.item() == CameraType.FISHEYE.value:
                    mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, None)
                else:
                    raise NotImplementedError("Only perspective and fisheye cameras are supported")
                data["mask"] = torch.from_numpy(mask)

            cached_eval.append(data)

            self.eval_dataset.cameras.fx[i] = float(K[0, 0])
            self.eval_dataset.cameras.fy[i] = float(K[1, 1])
            self.eval_dataset.cameras.cx[i] = float(K[0, 2])
            self.eval_dataset.cameras.cy[i] = float(K[1, 2])

        if cache_images_option == "gpu":
            for cache in cached_train:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
            for cache in cached_eval:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
        else:
            for cache in cached_train:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
            for cache in cached_eval:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()

        return cached_train, cached_eval

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        return smsDataset
    

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = list(range(len(self.eval_unseen_cameras)))
        data = copy(self.cached_eval)
        _cameras = copy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self):
        # TODO: fix this to be the resolution of the last image rendered
        return 800 * 800

    # @profile
    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        # if len(self.train_unseen_cameras) == 0:
        #     self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it

        # image_idx = self.train_unseen_cameras.pop()
        # print(image_idx)
        
        
        # start = time.time()
        data = copy(self.cached_train[image_idx])
        # import pdb; pdb.set_trace()
        data["image"] = data["image"].to(self.device)
        data["depth"] = data["depth"].to(self.device)
        # import pdb; pdb.set_trace()
        # end = time.time()
        # elapsed = str((end-start)*1e3)
        # print("copy time: "+ elapsed + "(ms)")

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx

        ########
        # CLIP #
        ########

        #Pick a random scale from min to max and then the clip features at that scale
        if self.use_clip:
            if step - self.lerf_step > 500 and len(self.clip_interpolator.data_dict[0].data) > image_idx:
                # print("Training CLIP")
                H, W = data["image"].shape[:2]
                H = H * self.dataparser_config.image_downscale_factor
                W = W * self.dataparser_config.image_downscale_factor
                scale = torch.rand(1).to(self.device)*(self.config.patch_tile_size_range[1]-self.config.patch_tile_size_range[0])+self.config.patch_tile_size_range[0]
                # import pdb; pdb.set_trace()
                # scale = torch.tensor([0.1]).to(self.device)*(self.config.patch_tile_size_range[1]-self.config.patch_tile_size_range[0])+self.config.patch_tile_size_range[0]
                self.curr_scale = scale
                scaled_height = H // self.config.clip_downscale_factor
                scaled_width = W // self.config.clip_downscale_factor

                detic_out = data["detic"]

                # bg_mask = None
                # components = detic_out["components"].to(int)
                # flat_components = detic_out["components"].reshape(-1).to(int)
                # # import pdb; pdb.set_trace()
                # if detic_out["masks"] is not None:
                #     bg_mask = ~(flat_components == 0)
                #     import matplotlib.pyplot as plt
                #     plt.imsave(f"mask{image_idx}.jpg", bg_mask.view(H, W).detach().cpu().numpy())
                
                # if bg_mask is not None:
                #     self.random_pixels = torch.arange(scaled_height*scaled_width) * bg_mask[::self.config.clip_downscale_factor ** 2]
                # else:

                # Construct mask negative pairs for contrastive loss
                with torch.no_grad():
                    # num_masks_to_sample = 10
                    masks = detic_out["masks_filtered"].squeeze(1) # masks: [num_masks, H, W]
                    # import pdb; pdb.set_trace()
                    assert len(masks) > 1, "Requires more than 1 mask"
                    assert masks.shape[1] == H and masks.shape[2] == W, "Mask shape mismatch"
                    assert len(masks.shape) == 3, "Instance masks should be a 2D tensor."
                    # perm = torch.randperm(len(masks))
                    # idx = perm[:num_masks_to_sample]
                    # num_masks = len(detic_out["masks_filtered"])

                    def create_mask_id(masks):
                        num_masks, H, W = masks.shape
                        mask_id = torch.full((H, W), -1, dtype=torch.int)  # Initialize with -1
                        for mask_idx in range(num_masks):
                            mask = masks[mask_idx]
                            mask_id[mask == 1] = mask_idx
                        return mask_id
                    
                    mask_id = F.interpolate(create_mask_id(masks).unsqueeze(0).unsqueeze(0).to(float), (scaled_height, scaled_width), mode = 'nearest').view(-1)
                    
                    batch_size = scaled_height * scaled_width
                    mask_id_flat = mask_id.view(-1)  # Shape: [batch_size]
                    labels1_expanded = mask_id_flat.unsqueeze(1).expand(-1, batch_size)
                    labels2_expanded = mask_id_flat.unsqueeze(0).expand(batch_size, -1)

                    mask_full_positive = labels1_expanded == labels2_expanded
                    mask_full_negative = ~mask_full_positive

                    # No need for block mask as we have a single image, but let's apply upper triangular mask to avoid double-counting
                    block_mask = torch.triu(torch.ones((batch_size, batch_size), dtype=torch.bool), diagonal=0).to(mask_id.device)

                    # Apply block mask to positive and negative masks
                    # valid_positive_mask = mask_full_positive * block_mask
                    valid_negative_mask = mask_full_negative * block_mask

                    # Exclude -1 pairs
                    # valid_positive_mask = valid_positive_mask & (mask_id_flat.unsqueeze(1) != -1) & (mask_id_flat.unsqueeze(0) != -1)
                    valid_negative_mask = valid_negative_mask & (mask_id_flat.unsqueeze(1) != -1) & (mask_id_flat.unsqueeze(0) != -1)

                    # # Example features (replace with actual computation)
                    # feature_dim = 128
                    # instance = torch.randn(batch_size, feature_dim, device=mask_id.device)

                    # Compute contrastive loss for negative pairs
                    # margin = 1.0
                    
                    mask = torch.where(valid_negative_mask)
                    # import pdb; pdb.set_trace()
                    if len(mask[0]) < 4000:
                        samp = len(mask[0])
                    else:
                        samp = 4000
                    random_perm = torch.randperm(len(mask[0]))[:samp]
                    mask = (mask[0][random_perm], mask[1][random_perm])
                    data["instance_masks"] = mask

                self.random_pixels = torch.randperm(scaled_height*scaled_width)[:int((scaled_height*scaled_height)*0.25)]

                x = torch.arange(0, scaled_width*self.config.clip_downscale_factor, self.config.clip_downscale_factor).view(1, scaled_width, 1).expand(scaled_height, scaled_width, 1)
                y = torch.arange(0, scaled_height*self.config.clip_downscale_factor, self.config.clip_downscale_factor).view(scaled_height, 1, 1).expand(scaled_height, scaled_width, 1)
                image_idx_tensor = torch.ones(scaled_height, scaled_width, 1)*image_idx
                positions = torch.cat((image_idx_tensor, y, x), dim=-1).view(-1, 3).to(int)

                # data["clip"], data["clip_scale"] = self.clip_interpolator(positions, scale)
                # image_encoder = self.clip_interpolator.data_dict[0].model
                # image_encoder.set_positives("white")  # some generic color that would be easy to identify...?
                # import matplotlib.pyplot as plt
                # plt.imsave("foo.png", image_encoder.get_relevancy(data['clip'], 0)[:, 0].view(scaled_height, scaled_width).detach().cpu().numpy())

                positions = positions[self.random_pixels]
                # import pdb; pdb.set_trace()
                with torch.no_grad():
                    # import pdb; pdb.set_trace()
                    # len(self.clip_interpolator.data_dict[0].data)
                    data["clip"], data["clip_scale"] = self.clip_interpolator(positions, scale) # [0], self.clip_interpolator(positions, scale)[1]
                    # data["clip"], data["clip_scale"] = self.clip_interpolator(positions)[0], self.clip_interpolator(positions)[1]
                    # data["dino"] = self.dino_dataloader(positions)
                # import pdb; pdb.set_trace()
                
                camera.metadata["clip_downscale_factor"] = self.config.clip_downscale_factor
                # import matplotlib.pyplot as plt
        # print('>>>', data.keys())
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = smsDataloader(self.train_dataset)
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        raise NotImplementedError


    def add_image(self, img: torch.tensor, cam: Cameras = None):
        """
        Adds a new image to the datamanager
        1. add the actual image data
        2. make sure pixel sampling works on that                        (Should work because we override the __getitem__ function in lll_dataset)
        3. add lerf dino+clip features                                   (Justin)
        4. reset camera param for optimization                           (I think we should do this in trainer on the image callback)
        5. make sure we set the mask for the image we just added         (We should handle masks in the pipeline because adding one image requires adding a bunch of masks)
        """
        # ----------------- Handling the lerf features ----------------
        self.clip_interpolator.add_images(img.unsqueeze(0))
    
    # def detic_image(self, img: torch.tensor):
    #     # img = torch.from_numpy(self.antialiased_downres(img, downscale_factor, True)).float()
    #     # self.train_dataset.add_image(img,depth,cam)
    #     # self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        
    #     # data = self.train_dataset[len(self.train_dataset)-1]
    #     # self.cached_train.append(data)
    #     detic_outouts = self.detic(img)

    #     return detic_outouts


    def antialiased_downres(self, image, factor, is_color_image=False):
        assert factor % 2 == 0, "factor must be a multiple of 2"
        
        kernel_size = 2 * factor - 1
        sigma = factor / 2

        torch_image = image

        torch_image = torch_image.unsqueeze(0).unsqueeze(0)
        blurred_image = cv2.GaussianBlur(np.array(torch_image),(kernel_size,kernel_size),sigma)
        if is_color_image:
            blurred_image = blurred_image.squeeze(0).transpose(0, 3, 1, 2)
        else:
            blurred_image = blurred_image
        blurred_image = torch.from_numpy(blurred_image).float()
        downsampled = F.interpolate(blurred_image, size=(image.shape[0] // factor, image.shape[1] // factor), mode='bilinear' ,antialias=True).numpy()
        if is_color_image:
            downsampled = downsampled.transpose(0, 2, 3, 1).squeeze(0)
        else:
            downsampled = downsampled.squeeze(0).squeeze(0)

        return downsampled

    # @profile
    def process_image(self, img:torch.Tensor, depth:torch.Tensor, cam: Cameras, clip, dino, downscale_factor = 1):
        # ----------------- Handling the IMAGE ----------------
        # raise NotImplementedError

        # DETIC
        detic_start_time = time.time()
        d_img = img.numpy().copy()
        d_img *= 255 # or any coefficient
        d_img = d_img.astype(np.uint8)
        detic_outputs = self.detic.predict(d_img)
        detic_end_time = time.time()
        detic_time = detic_end_time - detic_start_time
        print("Detic time: ", detic_time)

        # Antialiased downresolution
        img = torch.from_numpy(self.antialiased_downres(img, downscale_factor, True)).float()
        self.train_dataset.add_image(img,depth,cam, detic_outputs)
        self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        
        data = self.train_dataset[len(self.train_dataset)-1]
        self.cached_train.append(data)

    def add_to_clip(self, clip, step):
        if clip is not None: 
            if self.use_clip == False:
                print("Using CLIP")
                self.lerf_step = step
                print("Step recorded as step: " + str(step))
                self.use_clip = True
            for i, tr in enumerate(self.clip_interpolator.tile_sizes):
                clip[i] = clip[i].to(self.device)
                if self.clip_interpolator.data_dict[i].data is not None:
                    self.clip_interpolator.data_dict[i].data = torch.cat([self.clip_interpolator.data_dict[i].data, clip[i]])
                else:
                    self.clip_interpolator.data_dict[i].data = clip[i]
            # self.clip_interpolator.data_dict[0].model.set_positives(["tissue"])
            # scaled_height = 120
            # scaled_width = 212
            # x = torch.arange(0, scaled_width*4, 4).view(1, scaled_width,1).expand(scaled_height, scaled_width, 1)
            # y = torch.arange(0, scaled_height*4, 4).view(scaled_height, 1, 1).expand(scaled_height,scaled_width,1)
            # image_idx = 0
            # image_idx_tensor = torch.ones(scaled_height, scaled_width, 1)*image_idx
            # positions = torch.cat((image_idx_tensor, y, x), dim=-1).view(-1, 3).to(int)
            # scale = torch.rand(1).to(self.device)*(self.config.patch_tile_size_range[1]-self.config.patch_tile_size_range[0])+self.config.patch_tile_size_range[0]
            # data = copy(self.cached_train[image_idx])
            # data["clip"], data["clip_scale"] = self.clip_interpolator(positions)[0], self.clip_interpolator(positions)[1]
            # probs = self.clip_interpolator.data_dict[0].model.get_relevancy(data["clip"].view(-1, self.clip_interpolator.data_dict[0].model.embedding_dim), 0)
            # from nerfstudio.utils.colormaps import apply_colormap
            # color = apply_colormap(probs[..., 0:1])
            # color = color.reshape([120,212,3])
            # import matplotlib.pyplot as plt
            # plt.imshow(color.cpu().numpy())
            # plt.savefig(f"relevancy_{self.clip_interpolator.data_dict[0].model.positives}.png")
            # import pdb; pdb.set_trace()
        else:
            self.use_clip = False

    def detic_img(self, img):
        return self.detic(img)


    # def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
    #     """Returns the next batch of data from the train dataloader."""
    #     # raise NotImplementedError
    #     self.train_count += 1
    #     image_batch = next(self.iter_train_image_dataloader)
    #     assert self.train_pixel_sampler is not None
    #     batch = self.train_pixel_sampler.sample(image_batch)
    #     ray_indices = batch["indices"]
    #     ray_bundle = self.train_ray_generator(ray_indices)
    #     # batch["clip"], clip_scale = self.clip_interpolator(ray_indices)
    #     # batch["dino"] = self.dino_dataloader(ray_indices)
    #     # ray_bundle.metadata["clip_scales"] = clip_scale
    #     # # assume all cameras have the same focal length and image width
    #     ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
    #     ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
    #     ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
    #     ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
    #     return ray_bundle, batch

