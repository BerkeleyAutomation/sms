# from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import moviepy.editor as mpy
import wandb
from sam2.build_sam import build_sam2_camera_predictor
from sms.tracking.observation import PosedObservation, Frame
from torchvision.transforms.functional import to_pil_image
from sms.model.sms_gaussian_splatting import smsGaussianSplattingModel
import time
import cv2

da_image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
checkpoint = "/home/lifelong/sms/sms/data/utils/segment-anything-2-real-time/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
sam_pred = build_sam2_camera_predictor(model_cfg, checkpoint)
# da_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
# da_model.to('cuda')
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
    
def init_sam2(observation: PosedObservation, model: smsGaussianSplattingModel):
    assert sam_pred is not None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        frame_idx = 0
        sam_pred.load_first_frame(to_pil_image(observation.frame.rgb.permute(2,0,1)))
        for obj_id in range(len(observation.roi_frames)):
            outputs = model.get_outputs(observation._original_camera, tracking=True, obj_id=obj_id, BLOCK_WIDTH=8)
            object_mask = outputs["accumulation"] > 0.9
            points = sample_pixels_from_mask(object_mask.detach().cpu().numpy().squeeze(-1), num_samples=10) # Play around with num samples maybe make it proportional to mask area
            labels = np.ones((len(points),), dtype=np.int32)

            _, _, out_mask_logits = sam_pred.add_new_points(
                frame_idx = frame_idx,
                obj_id = obj_id,
                points = points,
                labels = labels
                )
        observation._obj_masks = out_mask_logits > 0.0
        if len(observation.roi_frames) > 0:
            for obj_id in range(len(observation.roi_frames)):
                frame = observation.roi_frames[obj_id]
                xmin, xmax, ymin, ymax = frame.xmin, frame.xmax, frame.ymin, frame.ymax
                observation.roi_frames[obj_id].obj_mask = (out_mask_logits > 0.0)[obj_id].squeeze(0)[ymin:ymax, xmin:xmax]

def propogate_sam2(observation: PosedObservation):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        start_time = time.time()
        _, out_mask_logits = sam_pred.track(to_pil_image(observation.frame.rgb.permute(2,0,1)))
        print(f"Time taken to propogate SAM2 masks {time.time()-start_time}")
        observation._obj_masks = out_mask_logits > 0.0
        if len(observation.roi_frames) > 0:
            for obj_id in range(len(observation.roi_frames)):
                frame = observation.roi_frames[obj_id]
                xmin, xmax, ymin, ymax = frame.xmin, frame.xmax, frame.ymin, frame.ymax
                observation.roi_frames[obj_id].obj_mask = (out_mask_logits > 0.0)[obj_id].squeeze(0)[ymin:ymax, xmin:xmax]
                
def sample_pixels_from_mask(mask: np.ndarray, num_samples: int) -> np.ndarray:
    """
    mask (np.ndarray): A boolean mask of shape [H, W].
    num_samples (int): The number of pixels to sample.

    Returns:
    np.ndarray: An array of shape [N, 2] containing the (x, y) coordinates of the sampled pixels.
    """
    
    indices = np.argwhere(mask)

    sampled_indices = indices[np.random.choice(indices.shape[0], num_samples, replace=False)]

    return sampled_indices.astype(np.float32)[...,::-1].copy()


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined