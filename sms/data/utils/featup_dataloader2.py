import torch
from sms.data.utils.dino_extractor import ViTExtractor
from sms.data.utils.feature_dataloader2 import FeatureDataloader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import torchvision.transforms as T
from featup.util import norm, unnorm
from featup.plotting import plot_feats, plot_lang_heatmaps
from tqdm import tqdm
import time
from torchvision.transforms.functional import resize

'''
https://github.com/mhamilton723/FeatUp
'''
import gc

def tensors_on_gpu():
    # This will hold all tensors found in GPU memory
    gpu_tensors = []

    # Iterate over all objects tracked by the garbage collector
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                gpu_tensors.append(obj)
        except:
            # Handle any object that can't be processed
            pass
    
    return gpu_tensors

# # Example usage
# x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
# y = torch.tensor([4.0, 5.0, 6.0], device='cpu')
# z = torch.tensor([7.0, 8.0, 9.0], device='cuda')

# gpu_tensors = tensors_on_gpu()
# for tensor in gpu_tensors:
#     print(tensor)

def get_img_resolution(H, W, max_size = 1050, p = None, downsample = None):
    if downsample:
            if p is not None:
                new_H = ((H//downsample)//p)*p
                new_W = ((W//downsample)//p)*p
            else:
                new_H = H//downsample
                new_W = W//downsample
    elif H<W:
        new_W = max_size
        new_H = (int((H/W)*max_size)//p)*p
    else:
        new_H = max_size
        new_W = (int((W/H)*max_size)//p)*p
    return new_H, new_W

class FeatupDataloader(FeatureDataloader):

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
        pca_dim: int = 64,
    ):
        assert "model_type" in cfg
        assert "image_shape" in cfg
        
        if cfg["model_type"] == "dinov2":
            self.model_type = "dinov2"
            use_norm = True
            self.upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
            self.downresolution = get_img_resolution(cfg['image_shape'][0], cfg['image_shape'][1], downsample=3, p=14)
            self.feat_dim = 384
            self.preprocess = T.Compose([
                T.Resize(self.downresolution),
                norm
            ])
            cfg["output_res"] = list(self.downresolution)
            
        elif cfg["model_type"] == "clip":
            import pdb; pdb.set_trace()
            self.model_type = "clip"
            use_norm = True
            self.upsampler = torch.hub.load("mhamilton723/FeatUp", 'clip', use_norm=use_norm).to(device)
            self.downresolution = get_img_resolution(cfg['image_shape'][0], cfg['image_shape'][1], downsample = 4)
            self.feat_dim = 512
            self.preprocess = T.Compose([
                T.Resize(self.downresolution),
                norm
            ])
            cfg["output_res"] = list(self.downresolution)
        
        self.device = device
        self.pca_dim = pca_dim
        super().__init__(cfg, device, image_list, cache_path)
        print(f"{self.model_type} data shape: {self.data.shape}")
    
    
    def create(self, image_list):
        self.data = self.get_feats(image_list)
        data_shape = self.data.shape
        if self.model_type == "dinov2":
            assert self.data.shape[-1] == 384
            if self.pca_dim != data_shape[-1]:
                print("Computing PCA")
                self.pca_matrix = torch.pca_lowrank(self.data.view(-1, data_shape[-1]), q=self.pca_dim,niter=20)[2]
                self.data = torch.matmul(self.data.view(-1, data_shape[-1]), self.pca_matrix).reshape((*data_shape[:-1], self.pca_dim))
            else:
                self.pca_matrix = torch.eye(data_shape[-1])
        elif self.model_type == 'clip':
            assert self.data.shape[-1] == 512
    
    def load(self):
        super().load()
        if self.model_type == "dinov2":
            cache_pca_path = self.cache_path.parent / ("pca.npy")
            self.pca_matrix = torch.from_numpy(np.load(cache_pca_path)).to(self.device)
        elif self.model_type == 'clip':
            cache_pca_path = self.cache_path.parent / ("clip.npy")
            self.clip_batch = torch.from_numpy(np.load(cache_pca_path)).to(self.device)

    def save(self):
        super().save()
        if self.model_type == "dinov2":
            cache_pca_path = self.cache_path.parent / ("pca.npy")
            np.save(cache_pca_path, self.pca_matrix.cpu().numpy())
        elif self.model_type == 'clip':
            cache_clip_path = self.cache_path.parent / ("clip.npy")
            np.save(cache_clip_path, self.data.cpu().numpy())

    def get_feats(self,image_list, keep_cuda=False):
        
        preproc_image_lst = self.preprocess(image_list).to(self.device)
        featup_embeds = []
        
        # for image in tqdm(preproc_image_lst, desc=self.model_type, total=len(image_list), leave=False):
        for image in preproc_image_lst:
            with torch.no_grad():
                image = image.unsqueeze(0)
                # Assert [B, C, H, W]
                assert image.shape[0] == 1, "Batch size must be 1"
                assert image.shape[1] == 3, "Image must have 3 channels"
                if keep_cuda:
                    featup_embeds.append(self.upsampler(image).permute(0, 2, 3, 1))
                    del image
                    torch.cuda.empty_cache()
                else:
                    featup_embeds.append(resize(self.upsampler(image).detach().cpu(), self.downresolution).permute(0, 2, 3, 1))
                    del image
                    torch.cuda.empty_cache()
                
        return torch.stack(featup_embeds, dim=0).squeeze(1)
    
    def get_pca_feats(self,image_list, keep_cuda = True):
        feats = self.get_feats(image_list, keep_cuda=keep_cuda)
        data_shape = feats.shape
        pca_feats = torch.matmul(feats.view(-1, data_shape[-1]), self.pca_matrix.to(feats)).reshape((*data_shape[:-1], self.pca_dim))
        return pca_feats
    
    def get_full_img_feats(self, img_ind) -> torch.Tensor:
        """
        returns BxHxWxC
        """
        return self.data[img_ind].to(self.device)
    
    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)

    def get_full_img_feats(self, img_ind) -> torch.Tensor:
        """
        returns BxHxWxC
        """
        return self.data[img_ind].to(self.device)
