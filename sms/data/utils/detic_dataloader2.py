# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import json
import os
import pickle
import random
from typing import List, Dict, Tuple
import sys
import time
import cv2
from sms.data.utils.feature_dataloader2 import FeatureDataloader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from PIL import Image
import torch
from tqdm import tqdm

# Change the current working directory to 'Detic'
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path+'/Detic')

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
sys.path.insert(0, 'third_party/CenterNet2/')

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries 
from sms.data.utils.Detic.detic.modeling.text.text_encoder import build_text_encoder
from collections import defaultdict
from centernet.config import add_centernet_config
from sms.data.utils.Detic.detic.config import add_detic_config
from sms.data.utils.Detic.detic.modeling.utils import reset_cls_test
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
from segment_anything import sam_model_registry, SamPredictor

class DeticDataloader(FeatureDataloader):
    def __init__(
            self,
            cfg: dict,
            device: torch.device,
            image_list: torch.Tensor = None,
            cache_path: str = None,
    ):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = cfg["sam"]
        # image_list: torch.Tensor = None,
        self.outs = [],
        super().__init__(cfg, device, image_list, cache_path)
    
    def create(self, image_list = None):
        # os.makedirs(self.cache_path, exist_ok=True)
        # Build the detector and download our pretrained weights
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
        # cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
        self.detic_predictor = DefaultPredictor(cfg)

        if self.sam == True:
            sam_checkpoint = "../sam_model/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            print('SAM + Detic on device: ', self.device)
            self.sam_predictor = SamPredictor(sam)
        self.text_encoder = build_text_encoder(pretrain=True)
        self.default_vocab()

        if image_list is not None:

            start_time = time.time()
            for idx, img in enumerate(tqdm(image_list, desc="Detic Detector", leave=False)):
                H, W = img.shape[-2:]
                img = (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)

                output = self.detic_predictor(img[:, :, ::-1])
                instances = output["instances"].to('cpu')

                boxes = instances.pred_boxes.tensor.numpy()

                masks = None
                components = torch.zeros(H, W)
                if self.sam:
                    if len(boxes) > 0:
                        # Only run SAM if there are bboxes
                        masks = self.SAM(img, boxes)
                        for i in range(masks.shape[0]):
                            if torch.sum(masks[i][0]) <= H*W/3.5:
                                components[masks[i][0]] = i + 1
                else:
                    masks = output['instances'].pred_masks.unsqueeze(1)
                    for i in range(masks.shape[0]):
                        if torch.sum(masks[i][0]) <= H*W/3.5:
                            components[masks[i][0]] = i + 1
                bg_mask = (components == 0).to(self.device)
                
                # Erode all masks using 3x3 kernel
                eroded_masks = torch.conv2d(
                (~masks).float().cuda(),
                torch.full((3, 3), 1.0).view(1, 1, 3, 3).to("cuda"),
                    padding=1,
                )
                eroded_masks = ~(eroded_masks >= 2) #.squeeze(1)
                # import pdb; pdb.set_trace()
                # Filter out small masks
                filtered_idx = []
                for i in range(len(masks)):
                    if masks[i].sum(dim=(1,2)) <= H*W/3.5:
                        filtered_idx.append(i)
                filtered_masks = torch.cat([eroded_masks[filtered_idx], bg_mask.unsqueeze(0).unsqueeze(0)], dim=0).cpu().numpy()

                outputs = {
                    # "vis": out,
                    "boxes": boxes,
                    "masks": masks,
                    "masks_filtered": filtered_masks,
                    # "class_idx": class_idx,
                    # "class_name": class_name,
                    # "clip_embeds": clip_embeds,
                    "components": components,
                    "scores" : output["instances"].scores,
                }
                # import pdb; pdb.set_trace()
                self.outs[0].append(outputs['masks_filtered'])
            # import pdb; pdb.set_trace()
            self.data = np.empty(len(image_list), dtype=object)
            self.data[:] = self.outs[0]
            # self.data = torch.stack(self.outs[0], dim=0)
            print("Detic batch inference time: ", time.time() - start_time)

    # Overridden load method
    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        # print(cache_info_path)
        # print(cache_info_path.exists())
        # import pdb; pdb.set_trace()
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            raise ValueError("Config mismatch")
        self.data = np.load(self.cache_path, allow_pickle=True)

    # Overridden save method
    def save(self):
        os.makedirs(self.cache_path.parent, exist_ok=True)
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        np.save(self.cache_path, self.data)

    def __call__(self, img_idx):
        return NotImplementedError

    def default_vocab(self):
        # detic_predictor = self.detic_predictor
        # Setup the model's vocabulary using build-in datasets
        BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
            'coco': 'datasets/metadata/coco_clip_a+cname.npy',
        }

        BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }

        vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
        self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
        classifier = BUILDIN_CLASSIFIER[vocabulary]

        num_classes = len(self.metadata.thing_classes)
        reset_cls_test(self.detic_predictor.model, classifier, num_classes)

    def get_clip_embeddings(self, vocabulary, prompt='a '):
        self.text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    # def SAM_predictors(self, device):
    #     sam_checkpoint = "../sam_model/sam_vit_h_4b8939.pth"
    #     model_type = "vit_h"
    #     device = device
    #     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    #     sam.to(device=device)
    #     self.sam_predictor = SamPredictor(sam)
    
    def SAM(self, im, boxes, class_idx = None, metadata = None):
        self.sam_predictor.set_image(im)
        input_boxes = torch.tensor(boxes, device=self.sam_predictor.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(input_boxes, im.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks

    def visualize_detic(self, output):
        output_im = output.get_image()[:, :, ::-1]
        cv2.imshow("Detic Predictions", output_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def custom_vocab(detic_predictor, classes):
    #     vocabulary = 'lvis'
    #     metadata = MetadataCatalog.get("__unused2")
    #     metadata.thing_classes = classes # Change here to try your own vocabularies!
    #     classifier = get_clip_embeddings(metadata.thing_classes)
    #     num_classes = len(metadata.thing_classes)
    #     reset_cls_test(detic_predictor.model, classifier, num_classes)

    #     # Reset visualization threshold
    #     output_score_threshold = 0.3
    #     for cascade_stages in range(len(detic_predictor.model.roi_heads.box_predictor)):
    #         detic_predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
    #     return metadata


    def predict(self, im):
        if im is None:
            print("Error: Unable to read the image file")

        H, W = im.shape[:2]

        # Run model and show results
        start_time = time.time()
        output = self.detic_predictor(im[:, :, ::-1])  # Detic expects BGR images.
        print("Inference time: ", time.time() - start_time)
        v = Visualizer(im, self.metadata)
        out = v.draw_instance_predictions(output["instances"].to('cpu'))
        instances = output["instances"].to('cpu')
        boxes = instances.pred_boxes.tensor.numpy()
        # class_idx = instances.pred_classes.numpy()
        # class_name = [self.metadata.thing_classes[idx] for idx in class_idx]
        # clip_embeds = self.get_clip_embeddings(class_name)

        masks = None
        components = torch.zeros(H, W)
        if self.sam:
            if len(boxes) > 0:
                # Only run SAM if there are bboxes
                masks = self.SAM(im, boxes)
                for i in range(masks.shape[0]):
                    if torch.sum(masks[i][0]) <= H*W/3.5:
                        components[masks[i][0]] = i + 1
        else:
            masks = output['instances'].pred_masks.unsqueeze(1)
            for i in range(masks.shape[0]):
                if torch.sum(masks[i][0]) <= H*W/3.5:
                    components[masks[i][0]] = i + 1
        bg_mask = (components == 0).to(self.device)

        # Filter out small masks
        filtered_idx = []
        for i in range(len(masks)):
            if masks[i].sum(dim=(1,2)) <= H*W/3.5:
                filtered_idx.append(i)
        filtered_masks = torch.cat([masks[filtered_idx], bg_mask.unsqueeze(0).unsqueeze(0)], dim=0)

        # invert_masks = ~filtered_masks
        # # erode all masks using 3x3 kernel
        # eroded_masks = torch.conv2d(
        #     invert_masks.float(),
        #     torch.full((3, 3), 1.0).view(1, 1, 3, 3).to("cuda"),
        #     padding=1,
        # )
        # filtered_masks = ~(eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

                        
        outputs = {
            "vis": out,
            "boxes": boxes,
            "masks": masks,
            "masks_filtered": filtered_masks,
            # "class_idx": class_idx,
            # "class_name": class_name,
            # "clip_embeds": clip_embeds,
            "components": components,
            "scores" : output["instances"].scores,
        }
        return outputs


    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


    def visualize_output(self, im, masks, input_boxes, classes, image_save_path, mask_only=False):
        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        if not mask_only:
            for box, class_name in zip(input_boxes, classes):
                self.show_box(box, plt.gca())
                x, y = box[:2]
                plt.gca().text(x, y - 5, class_name, color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='green', edgecolor='green', alpha=0.5))
        plt.axis('off')
        plt.savefig(image_save_path)
        #plt.show()


    def generate_colors(self, num_colors):
        hsv_colors = []
        for i in range(num_colors):
            hue = i / float(num_colors)
            hsv_colors.append((hue, 1.0, 1.0))

        return [mcolors.hsv_to_rgb(color) for color in hsv_colors]


    # def main(args):

    #     # We are one directory up in Detic.
    #     image_path = os.path.join("..", args.image_path)
        
    #     image = Image.open(image_path)
    #     image = np.array(image, dtype=np.uint8)

    #     detic_predictor = DETIC_predictor()
    #     metadata = default_vocab(detic_predictor, args.classes)

    #     boxes, class_idx = Detic(image, metadata, detic_predictor)
    #     assert len(boxes) > 0, "Zero detections."
    #     masks = SAM(image, boxes, class_idx, metadata, sam_predictor)

    #     # Save detections as a png.
    #     # Add "_bbox" before the suffix.
    #     image_save_path = image_path.split(".")
    #     image_save_path[-2] += "_bbox"
    #     image_save_path = ".".join(image_save_path)
    #     classes = [metadata.thing_classes[idx] for idx in class_idx]
    #     visualize_output(image, masks, boxes, classes, image_save_path)

    #     # Save only segmentation without bounding box as a separate image.
    #     # Add "_segm" before the suffix.
    #     image_save_path = image_path.split(".")
    #     image_save_path[-2] += "_segm"
    #     image_save_path = ".".join(image_save_path)
    #     classes = [metadata.thing_classes[idx] for idx in class_idx]
    #     visualize_output(image, masks, boxes, classes, image_save_path, mask_only=True)

    #     # Save detections as a pickle.
    #     pickle_save_path = image_path.split(".")
    #     pickle_save_path[-2] += "_segm"
    #     pickle_save_path[-1] = "pkl"
    #     pickle_save_path = ".".join(pickle_save_path)

    #     with open(pickle_save_path, "wb") as f:
    #         pickle.dump({
    #             "masks": masks.cpu().numpy(),
    #             "boxes": boxes,  # y_min, x_min, y_max, x_max
    #             "classes": classes
    #         }, f)
