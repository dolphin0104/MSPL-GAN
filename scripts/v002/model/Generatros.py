import math
import functools
import numpy as np
import torch
import torch.nn as nn
from model.modules import basicblocks as B
from model.modules.model_utils import *

class FocusNet(nn.Module):
    def __init__(self, block_type, n_blocks, in_nc=3, out_nc=3, n_feat=64, gc=None,
            norm_type=None, act_type='relu', mode='CNA', res_scale=0.2,
            scale=None):
        super(FocusNet, self).__init__()
        if block_type == 'resnet':
            body = [B.ResNetBlock(n_feat, n_feat, n_feat, norm_type=norm_type, 
                act_type=act_type, mode=mode, res_scale=res_scale) 
                for _ in range(n_blocks)]
        elif block_type == 'rrdb':
            body = [B.RRDB(n_feat, kernel_size=3, gc=32, stride=1, 
                bias=True, pad_type='zero', norm_type=norm_type, 
                act_type=act_type, mode='CNA') for _ in range(n_blocks)] 
        else: 
            raise NotImplementedError('block_type \
                [{:s}] not recognized'.format(block_type))        
        self.scale = scale
        self.head_conv = B.conv_block(in_nc, n_feat, 
            kernel_size=3, norm_type=None, act_type=None)
        self.body = B.sequential(*body)        
        self.body_conv = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=norm_type, act_type=None, mode=mode)
        
        if scale:
            n_upscale = int(math.log(scale, 2))
            upsampler = [B.pixelshuffle_block(n_feat, n_feat, act_type=act_type) 
                for _ in range(n_upscale)]
            self.upsampler = B.sequential(*upsampler)
        self.tail_conv0 = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=None, act_type=act_type)
        self.tail_conv1 = B.conv_block(n_feat, out_nc, kernel_size=3, 
            norm_type=None, act_type=None)

    def forward(self, x):
        res = self.head_conv(x)
        x = self.body(res)
        x = self.body_conv(x)
        x = x + res
        if self.scale:
            x= self.upsampler(x)
        x = self.tail_conv0(x)
        x = self.tail_conv1(x)       
        return x

class FocusResNet(nn.Module):
    def __init__(self, n_blocks, in_nc=3, out_nc=3, n_feat=64, gc=None,
            norm_type=None, act_type='relu', mode='CNA', res_scale=0.2):
        super(FocusResNet, self).__init__()
        self.head_conv = B.conv_block(in_nc, n_feat, 
            kernel_size=3, norm_type=None, act_type=None)

        body = [B.ResNetBlock(n_feat, n_feat, n_feat, norm_type=norm_type, 
            act_type=act_type, mode=mode, res_scale=res_scale) for _ in range(n_blocks)]
        self.body = B.sequential(*body)        
        self.body_conv = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=norm_type, act_type=None, mode=mode)
        
        self.tail_conv0 = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=None, act_type=act_type)
        self.tail_conv1 = B.conv_block(n_feat, out_nc, kernel_size=3, 
            norm_type=None, act_type=None)

    def forward(self, x):
        res = self.head_conv(x)
        x = self.body(res)
        x = self.body_conv(x)
        x = x + res
        x = self.tail_conv0(x)
        x = self.tail_conv1(x)       
        return x

class FocusRRDBNet(nn.Module):
    def __init__(self, n_blocks, in_nc=3, out_nc=3, n_feat=64, gc=32, 
            norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=None):
        super(FocusRRDBNet, self).__init__()        
        self.head_conv = B.conv_block(in_nc, n_feat, 
            kernel_size=3, norm_type=None, act_type=None)
        
        body = [B.RRDB(n_feat, kernel_size=3, gc=32, stride=1, 
            bias=True, pad_type='zero', norm_type=norm_type, 
            act_type=act_type, mode='CNA') for _ in range(n_blocks)]
        self.body = B.sequential(*body)        
        self.body_conv = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=norm_type, act_type=None, mode=mode)

        self.tail_conv0 = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=None, act_type=act_type)
        self.tail_conv1 = B.conv_block(n_feat, out_nc, kernel_size=3, 
            norm_type=None, act_type=None)

    def forward(self, x):
        res = self.head_conv(x)
        x = self.body(res)
        x = self.body_conv(x)
        x = x + res
        x = self.tail_conv0(x)
        x = self.tail_conv1(x)       
        return x
