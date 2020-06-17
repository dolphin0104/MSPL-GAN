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


class Testset(data.Dataset):
    """Dataset class Test dataset."""
    def __init__(self, data_dir, dataset_type, dataset_name):        
        
        self.data_dir = data_dir   
        self.dataset_type = dataset_type       
        self.dataset_name = dataset_name
        self.rgb_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) 

        self.items = self._set_filelist()

    def _set_filelist(self):

        if self.dataset_type == 'CVPR18':
            gt_dir = os.path.join(self.data_dir, 'final_{}_gt'.format(self.dataset_name))       
            blur_dir = os.path.join(self.data_dir, 'final_{}_blur'.format(self.dataset_name)) 

        else:
            gt_dir = os.path.join(self.data_dir, '{}_gt'.format(self.dataset_name))      
            blur_dir = os.path.join(self.data_dir, '{}_blur'.format(self.dataset_name)) 

        items = []

        for blur_img in os.listdir(blur_dir):
            blur_path = os.path.join(blur_dir, blur_img)
            blur_filename = os.path.splitext(os.path.split(blur_path)[-1])[0]
            
            for gt_img in os.listdir(gt_dir):
                gt_path = os.path.join(gt_dir, gt_img)
                gt_filename = os.path.splitext(os.path.split(gt_path)[-1])[0]

                if gt_filename == blur_filename[:-15]:
                    items.append([gt_path, gt_filename, blur_path, blur_filename])
                       
        assert len(items) == len(os.listdir(blur_dir)), "IMAGE & MASK LIST NUM are Different!!!"

        return items


    def __getitem__(self, idx):
        gt_path, gt_filename, blur_path, blur_filename = self.items[idx]
                
        gt_img = Image.open(gt_path)
        blur_img = Image.open(blur_path)        
       
        # to tensor
        gt_img = self.rgb_to_tensor(gt_img)
        blur_img = self.rgb_to_tensor(blur_img)
    
        return gt_img, gt_filename, blur_img, blur_filename
    

    def __len__(self):
        return len(self.items)