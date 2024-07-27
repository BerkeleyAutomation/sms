from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import moviepy.editor as mpy
import wandb

da_image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
da_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
da_model.to('cuda')
def get_depth(img: Union[torch.tensor,np.ndarray]):
    assert img.shape[2] == 3
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
    image = Image.fromarray(img)

    # prepare image for the model
    inputs = da_image_processor(images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].cuda()

    with torch.no_grad():
        outputs = da_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    return prediction.squeeze()

hand_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
hand_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
hand_model.to('cuda')
def get_hand_mask(img: Union[torch.tensor,np.ndarray]):
    assert img.shape[2] == 3
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
    image = Image.fromarray(img)

    # prepare image for the model
    inputs = hand_processor(images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].cuda()

    with torch.no_grad():
        outputs = hand_model(**inputs)

    # Perform post-processing to get panoptic segmentation map
    seg_ids = hand_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    hand_mask = (seg_ids == hand_model.config.label2id['person']).float()
    return hand_mask

def generate_videos(frames_dict, fps=30, config_path=None):
    import datetime
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for key in frames_dict.keys():
        frames = frames_dict[key]
        if len(frames)>1:
            if frames[0].max() > 1:
                frames = [f for f in frames]
            else:
                frames = [f*255 for f in frames]
        clip = mpy.ImageSequenceClip(frames, fps=fps)
        if config_path is None:
            clip.write_videofile(f"{timestr}/{key}.mp4", codec="libx264")
        else:
            path = config_path.joinpath(f"{timestr}")
            if not path.exists():
                path.mkdir(parents=True)
            clip.write_videofile(str(path.joinpath(f"{key}.mp4")), codec="libx264")
        try:
            wandb.log({f"{key}": wandb.Video(str(path.joinpath(f"{key}.mp4")))})
        except:
            pass
    return timestr
    