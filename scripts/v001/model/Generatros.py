import math
import functools
import numpy as np
import torch
import torch.nn as nn
from model.modules import basicblocks as B
from model.modules.model_utils import *

class FocusResNet(nn.Module):
    def __init__(self, model_type, in_nc, out_nc, n_feat, n_blocks, 
            norm_type=None, act_type='relu', mode='CNA', res_scale=0.2):
        super(FocusResNet, self).__init__()
        head_conv = B.conv_block(in_nc, n_feat, 
            kernel_size=3, norm_type=None, act_type=None)

        resnet_blocks = [B.ResNetBlock(n_feat, n_feat, n_feat, norm_type=norm_type, 
            act_type=act_type, mode=mode, res_scale=res_scale) for _ in range(n_blocks)]
        tail_conv = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=norm_type, act_type=None, mode=mode)
        
        recon_conv0 = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=None, act_type=act_type)
        recon_conv1 = B.conv_block(n_feat, out_nc, kernel_size=3, 
            norm_type=None, act_type=None)

        self.model = B.sequential(head_conv, 
            B.ShortcutBlock(B.sequential(*resnet_blocks, tail_conv)),
            recon_conv0, recon_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class FocusRRDBNet(nn.Module):
    def __init__(self, model_type, in_nc, out_nc, n_feat, n_blocks, gc=32, 
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(FocusRRDBNet, self).__init__()
        
        head_conv = B.conv_block(in_nc, n_feat, kernel_size=3, 
            norm_type=None, act_type=None)
        
        rb_blocks = [B.RRDB(n_feat, kernel_size=3, gc=32, stride=1, 
            bias=True, pad_type='zero', norm_type=norm_type, 
            act_type=act_type, mode='CNA') for _ in range(n_blocks)]

        tail_conv = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=norm_type, act_type=None, mode=mode)

        recon_conv0 = B.conv_block(n_feat, n_feat, kernel_size=3, 
            norm_type=None, act_type=act_type)
        recon_conv1 = B.conv_block(n_feat, out_nc, kernel_size=3, 
            norm_type=None, act_type=None)

        self.model = B.sequential(head_conv, 
            B.ShortcutBlock(B.sequential(*rb_blocks, tail_conv)),
            recon_conv0, recon_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
