import os
import numpy as np
import random
import cv2
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms

from model.generator import MSPL_Generator
from utils.misc import make_dir, write_log
from utils.visualize import denorm, tensor2img

# 1. Set Paths
MODEL_DIR = ''
INPUT_IMG_DIR = ''
OUTPUT_IMG_DIR = ''
make_dir(OUTPUT_IMG_DIR)

# 2. Set GPU or CPU 
device = torch.device('cuda') # if USE_GPU else device = torch.device('cpu')

# 2. Model Load
netG = MSPL_Generator(3, 3, 128, [4, 4, 4, 4]).to(device)
netG = nn.DataParallel(netG)

rgb_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

if isinstance(netG, nn.DataParallel):
    netG = netG.module
netG.load_state_dict(torch.load(MODEL_DIR), strict=strict)
netG.eval()

for imgs in os.listdir(INPUT_IMG_DIR):    
    img_path = os.path.join(INPUT_IMG_DIR, imgs)
    img_filename = os.path.splitext(os.path.split(img_path)[-1])[0]

    img_pil = Image.open(img_path)
    img_tensor = rgb_to_tensor(img_pil)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = img_tensor.to(device)

    out_list = netG(img_tensor)
    for i, out_tensor in enumerate(out_list):
        out_tensor = denorm(out_tensor)
        out_img = tensor2img(out_tensor)
        save_filename = os.path.join(OUTPUT_IMG_DIR, '{}_out{}.png'.format(img_filename, i))
        cv2.imwrite(save_filename, out_img)
    print("{} is done!".format(img_filename))
