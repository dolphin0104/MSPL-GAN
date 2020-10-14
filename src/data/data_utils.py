"""
Code Reference:
    https://github.com/ms-sharma/Adversarial-Semisupervised-Semantic-Segmentation
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import math
import random
import numpy as np


class OneHotEncode(object):
    """
        Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    """
    def __init__(self, n_classes=13):
        self.n_classes = n_classes

    def __call__(self,label):
        label_a = np.array(transforms.ToPILImage()(label.byte().unsqueeze(0)),np.uint8)
        ohlabel = np.zeros((self.n_classes,label_a.shape[0],label_a.shape[1])).astype(np.uint8)

        for c in range(self.n_classes):
            ohlabel[c:,:,:] = (label_a == c).astype(np.uint8)

        return torch.from_numpy(ohlabel)
    

class IgnoreLabelClass(object):
    """
        Convert a label for a class to be ignored to some other class
    """
    def __init__(self,ignore=255,base=0):
        self.ignore = ignore
        self.base = base

    def __call__(self,label):
        return Image.eval(label,lambda p: self.base if p == self.ignore else p)


class ToTensorLabel(object):
    """
        Take a Label as PIL.Image with 'P' mode and convert to Tensor
    """
    def __init__(self,tensor_type=torch.LongTensor):
        self.tensor_type = tensor_type

    def __call__(self,label):
        label = np.array(label,dtype=np.uint8)
        label = torch.from_numpy(label).type(self.tensor_type)

        return label