# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import json
import os
import pickle
import random
from typing import List, Dict, Tuple
import sys

import cv2
from sms.data.utils.feature_dataloader import FeatureDataloader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from PIL import Image
import torch

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
from detic.modeling.text.text_encoder import build_text_encoder
from collections import defaultdict
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches

class DeticDataloader():

    def create(self):
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
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

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

        # Run model and show results
        output = self.detic_predictor(im[:, :, ::-1])  # Detic expects BGR images.
        v = Visualizer(im, self.metadata)
        out = v.draw_instance_predictions(output["instances"].to('cpu'))
        instances = output["instances"].to('cpu')
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        # if visualize:
            # visualize_detic(out)
        return out, boxes, classes


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