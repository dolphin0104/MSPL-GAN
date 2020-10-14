import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import random
from data import joint_transforms, data_utils


class CelebA_HQ(data.Dataset):
    """Dataset class for the CelebA_HQ dataset."""
    def __init__(self, data_dir, n_classes=4, mode='train'):        
        assert mode in ['train', 'val', 'test']
        self.data_dir = data_dir
        self.mode = mode
        self.n_classes = n_classes
        self.img_list, self.mask_list = self._set_filelist()
              
        if mode == 'train':
            self.trans = joint_transforms.Compose([
                joint_transforms.ColorJitter(0.5, 0.5, 0.5),
                joint_transforms.RandomCrop(448),
                joint_transforms.Resize(128),
                joint_transforms.RandomHorizontallyFlip(),
                joint_transforms.RandomRotate(270),
            ])
        else:
            self.trans = joint_transforms.Compose([
                joint_transforms.Resize(128),
            ])

        self.rgb_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) 
    

    def _set_filelist(self):
        img_dir = os.path.join(self.data_dir, '{}_img'.format(self.mode))       
        mask_dir = os.path.join(self.data_dir, '{}_label_cls4'.format(self.mode))

        img_list = []
        mask_list = []

        for imgs in os.listdir(img_dir):
            img_path = os.path.join(img_dir, imgs)
            img_list.append(img_path)
            filename = os.path.splitext(os.path.split(img_path)[-1])[0]

            mask_path = os.path.join(mask_dir, filename+'.png')
            mask_list.append(mask_path)
        
        assert len(img_list) == len(mask_list), "IMAGE & MASK LIST NUM are Different!!!"

        #return img_list[:80], mask_list[:80]
        return img_list, mask_list


    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.mask_list[idx]
        filename = os.path.splitext(os.path.split(img_path)[-1])[0] 
        
        gt_img = Image.open(img_path)
        gt_segmap = Image.open(mask_path).convert('P')
        
        if gt_img.size != gt_segmap.size:
            w, h = gt_segmap.size
            gt_img = gt_img.resize((w, h), Image.BILINEAR)
        
        # augmentation
        gt_img, gt_segmap = self.trans(gt_img, gt_segmap)

        # to tensor
        gt_img = self.rgb_to_tensor(gt_img)
        gt_segmap = data_utils.ToTensorLabel()(gt_segmap)
        gt_onehot_segmap = data_utils.OneHotEncode(n_classes=self.n_classes)(gt_segmap).float() 
        
        return gt_img, gt_onehot_segmap, filename


    def __len__(self):
        return len(self.img_list)


#============================================================
# custom valdationset
#============================================================
class validface(data.Dataset):
    """Dataset class for the CelebA_HQ dataset."""
    def __init__(self, data_dir):        
        
        self.data_dir = data_dir             
        self.rgb_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) 

        self.items = self._set_filelist()

    def _set_filelist(self):
        gt_dir = os.path.join(self.data_dir, 'gt')       
        blur_dir = os.path.join(self.data_dir, 'blur')

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