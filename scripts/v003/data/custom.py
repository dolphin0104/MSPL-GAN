import os
import cv2
import numpy as np
import random
import torch
import torch.utils.data as data
from data import utils


class customDataset(data.Dataset):
    def __init__(self, img_dir, name=None,
                 img_size=144, patch_size=None, crop_size=128,
                 n_channels=3, rgb_range=1,
                 batch_size=16, test_every=1000, train=True):
        super(customDataset, self).__init__()
        self.name = name
        self.n_channels = n_channels
        self.rgb_range = rgb_range
        self.img_size = img_size
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.train = train
        # Get image path lists
        self.imgList = utils.get_paths_from_images(img_dir)
        # n_imgs = len(self.imgList)
        # if train:
        #     self.repeat = max((batch_size * test_every) // n_imgs, 1)
    
    def __getitem__(self, idx):
        img_gt, filename = self._load_file(idx)
        img_gt = self._preprocess(img_gt)
        img_gt = utils.set_channel([img_gt], self.n_channels)
        img_gt = utils.np2Tensor([img_gt], self.rgb_range)
        return img_gt, filename

    def _preprocess(self, img_gt):
        # resizing
        if self.img_size:
            img_gt = utils.img_resize(img_gt, size=self.img_size)

        if self.train:           
            if self.patch_size:
                img_gt = utils.get_patch(img_gt, size=self.patch_size)
            if self.crop_size:
                img_gt = utils.img_randomcrop(img_gt, size=self.crop_size)            
            img_gt = utils.augment([img_gt]) # random horizontal flip                
       
        return img_gt       
    
    def _load_file(self, idx):
        filename = os.path.splitext(os.path.split(self.imgList[idx])[-1])[0]
        img = cv2.imread(self.imgList[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, filename
    
    def __len__(self):
        return len(self.imgList)

    def _get_index(self, idx):
        return idx




    

