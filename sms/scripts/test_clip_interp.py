from data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from encoders.image_encoder import BaseImageEncoderConfig
from encoders.openclip_encoder import OpenCLIPNetworkConfig
from datasets import load_dataset
from torch.utils import data
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.colormaps import apply_colormap
import torch

def main():
    device = 'cuda:6'
    """The device to run on"""
    patch_tile_size_range: Tuple[int, int] = (0.08, 0.5)
    """The range of tile sizes to sample from for patch-based training"""
    patch_tile_size_res: int = 7
    """The number of tile sizes to sample from for patch-based training"""
    patch_stride_scaler: float = 0.5
    """The stride scaler for patch-based training"""
    network: BaseImageEncoderConfig = OpenCLIPNetworkConfig(device=device)
    """specifies the vision-language network config"""
    clip_downscale_factor: int = 1
    """The downscale factor for the clip pyramid"""

    clip_interpolator = PyramidEmbeddingDataloader(
        device=device,
        cfg={
            "tile_size_range": list(patch_tile_size_range),
            "tile_size_res": patch_tile_size_res,
            "stride_scaler": patch_stride_scaler,
            # "image_shape": [h,w],
        },
        model=network.setup()
    )
    image_encoder = clip_interpolator.model

    dataset = load_dataset("imagenet-1k")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    ### Load human readable labels dictionary from data/labels/imagenet1k_labels.txt
    with open("data/labels/imagenet1k_labels.txt", "r") as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    # List of "930: 'ear, spike, capitulum'," to dictionary
    labels = {int(label.split(":")[0]): label.split(":")[1].strip(",").strip().strip("'") for label in labels}
    # import pdb; pdb.set_trace()

    data = {}
    for i, batch in enumerate(dataset['train']):

        image = transform(batch['image'])
        try:
            clip_interpolator.generate_clip_interp(image)
        except:
            continue
        H, W = image.shape[1:]

        scale = torch.tensor(0.1).to(device)
        scaled_height = H//clip_downscale_factor
        scaled_width = W//clip_downscale_factor
        # random_pixels = torch.randperm(scaled_height*scaled_width)[:int((scaled_height*scaled_height)*0.5)]

        x = torch.arange(0, scaled_width*clip_downscale_factor, clip_downscale_factor).view(1, scaled_width, 1).expand(scaled_height, scaled_width, 1).to(device)
        y = torch.arange(0, scaled_height*clip_downscale_factor, clip_downscale_factor).view(scaled_height, 1, 1).expand(scaled_height, scaled_width, 1).to(device)
        image_idx_tensor = torch.zeros(scaled_height, scaled_width, 1).to(device)
        positions = torch.cat((image_idx_tensor, y, x), dim=-1).view(-1, 3).to(int)
        # positions = positions[random_pixels]
        with torch.no_grad():
            # data["clip"], data["clip_scale"] = clip_interpolator(positions, scale)[0], clip_interpolator(positions, scale)[1]
            data["clip"] = clip_interpolator(positions)[0]

        # import pdb; pdb.set_trace()

        positive = labels[batch["label"]].split(", ")
        # import pdb; pdb.set_trace()
        image_encoder.set_positives(positive)
        probs = image_encoder.get_relevancy(data["clip"].view(-1, image_encoder.embedding_dim), 0)
        color = apply_colormap(probs[..., 0:1])
        color = color.reshape([H,W,3])
        # Show image and heatmap side by side
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image.permute(1,2,0))
        ax[1].imshow(color.cpu().numpy())
        fig.suptitle(positive)
        plt.savefig(f"test_clip_interp_{i}_{positive}.png")

        if i == 100:
            break
    

if __name__ == '__main__':
    main()