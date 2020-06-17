import os
import numpy as np
import random
import math
import cv2
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils import data
from torchvision import transforms


def load_network(model, load_path, strict=True):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(torch.load(load_path), strict=strict)

    
def prepare(l, device, volatile=False):
    def _prepare(tensor): return tensor.to(device)           
    return [_prepare(_l) for _l in l]


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_log(log, save_path, save_name, refresh=False):
    #print(log)
    log_txt = os.path.join(save_path, '{}.txt'.format(save_name))
    open_type = 'a' if os.path.exists(log_txt) else 'w'
    log_file = open(log_txt, open_type)
    log_file.write(str(log))
    log_file.write('\n')
    if refresh:
        log_file.close()
        log_file = open(log_txt, 'a') 


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), is_img=True):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.detach().numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.detach().numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        if not is_img:
            img_np = (img_np + 128.0).round().clip(0, 255)
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)