import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from tqdm import tqdm, trange
import pickle
import PIL.Image as Image
import json
import random
import sys
# import clip
import PIL
import random
from sms.encoders.openclip_encoder import *


device = torch.device('cuda:0')
# clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
# tokenizer = clip.tokenize
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# _Tokenizer = _Tokenizer()
clipencoder = OpenCLIPNetworkConfig(device=device).setup()


text = "a person on a motorcycle rider jumps in the air"

# texts_token = tokenizer(text).to(device)
# text_feature = clip_model.encode_text(texts_token)

tokenizer = open_clip.get_tokenizer('ViT-B-32')
text_embed = clipencoder.model.encode_text(tokenizer(text).cuda())

text_embed /= text_embed.norm(dim=-1,keepdim=True)

path_pic = 'images/000000190756.jpg'  #Pictures from MSCOCO val.
image = Image.open(path_pic)
# image = preprocess(image).unsqueeze(0).to(device)
# image_features = clip_model.encode_image(image).float()
# image_features /= image_features.norm(dim=-1,keepdim=True)

clipimg = clipencoder.process(image).unsqueeze(0).cuda()
imgenc = clipencoder.model.encode_image(clipimg).float()

random_feature = torch.rand(1,512).to(device)
# compute cosine similarity in dim=1 
cos_2 = torch.nn.CosineSimilarity(dim=1) 
output_2 = cos_2(text_embed, imgenc) 
output_3 = cos_2(imgenc, random_feature)
output_4 = cos_2(random_feature,text_embed)
# display the output tensor 
print("\n\nComputed Cosine Similarity for Image and Text in dim=1: ", 
      output_2)
print("\n\nComputed Cosine Similarity for Image and Random in dim=1: ", 
      output_3)
print("\n\nComputed Cosine Similarity for Random and Text in dim=1: ", 
      output_4)