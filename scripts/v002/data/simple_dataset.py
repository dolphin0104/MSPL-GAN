import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
from data import data_utils

class simpleDataset(data.Dataset):
    def __init__(self,
        image_dir, n_data, train=True,
        image_size=(112, 112), n_channels=3, interp='cubic', rgb_range=1,
        ran_crop=True, crop_size=(112, 96),
        gen_lr=False, scale=None,
        add_blur=True, blur_type='G', blur_ksize=25, blur_val=100, 
        add_noise=True, noise_type='G', noise_val=0.01,
        is_augment=True, hflip=True, rot=False):
        super(simpleDataset, self).__init__()
        self.train = train
        self.image_size = image_size
        self.n_channels = n_channels
        self.interp = interp
        self.rgb_range = rgb_range
        self.gen_lr = gen_lr
        self.add_blur =  add_blur
        self.add_noise = add_noise
        self.ran_crop = ran_crop
        if gen_lr:
            self.scale = scale
        if add_blur:            
            self.blur_type = blur_type
            self.blur_ksize = blur_ksize
            self.blur_val = blur_val
        if add_noise:
            self.noise_type = noise_type
            self.noise_val = noise_val
        if is_augment:
            self.is_augment = is_augment
            self.hflip = hflip
            self.rot = rot
        if ran_crop:
            self.ran_crop = ran_crop
            self.crop_size = crop_size
        self.imgList = None
        # Get image path list
        self.imgList = data_utils.get_paths_from_images(image_dir)
        if n_data:
            self.data_set = []
            self.n_data = n_data
            self._set_list() 
            assert len(self.data_set) == self.n_data
        else: 
            self.n_data = len(self.imgList)               
            self.data_set = self.imgList
        
    def _set_list(self):
        random.seed(1234)
        random.shuffle(self.imgList)  
        for i, files in enumerate(self.imgList):
            if i < self.n_data:
                self.data_set.append(files)
    
    def __getitem__(self, idx):
        img_gt, filename = self._load_img(idx)
        img_gt, img_in = self._preprocess(img_gt)
        img_gt, img_in = data_utils.set_channel([img_gt, img_in], self.n_channels)
        img_gt, img_in = data_utils.np2Tensor([img_gt, img_in], self.rgb_range)
        return img_gt, img_in, filename
    
    def _load_img(self, idx):        
        filename = os.path.splitext(os.path.split(self.data_set[idx])[-1])[0]
        img = cv2.imread(self.data_set[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, filename
    
    def _preprocess(self, img_gt):
        if self.train:
            if self.image_size:
                img_gt = data_utils.img_resize(img_gt, self.image_size, self.interp)
            if self.gen_lr:
                img_in = data_utils.img_gen_lr(img_gt, self.scale, self.interp)
            else:
                img_in = np.copy(img_gt)
            if self.ran_crop:
                img_in = data_utils.get_randomcrop(img_in, self.crop_size)
            if self.is_augment:
                img_gt, img_in = data_utils.augment([img_gt, img_in], self.hflip, self.rot)
        else:
            if self.gen_lr:
                img_in = data_utils.img_gen_lr(img_gt, self.scale, self.interp)
            else:
                img_in = np.copy(img_gt)
        if self.add_blur:
            img_in = data_utils.add_blur(img_in, self.blur_type, 
                self.blur_ksize, self.blur_val)
        if self.add_noise:
            img_in = data_utils.add_noise(img_in, self.noise_type, self.noise_val)
        return img_gt, img_in

    def __len__(self):
        return len(self.data_set)
        
