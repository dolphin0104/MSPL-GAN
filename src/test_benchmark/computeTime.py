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

import time

def computeTime(model, device='cuda'):
    inputs = torch.randn(1, 3, 128, 128)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 10:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time: {:.3f}'.format(np.mean(time_spent)))
    
    
model = MSPL_Generator(3, 3, 128, [4, 4, 4, 4])
computeTime(model)