import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def get_norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


## Channel Attention (CA) Layer
class SEModule(nn.Module):
    def __init__(self, n_feats=64, reduction=16):
        super(SEModule, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.channel_attention = nn.Sequential(
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.channel_attention(y)
        return x * y
    

class ResBlock(nn.Module):
    """ ResNet Block composed of 2 conv blocks"""
    def __init__(self, 
                 n_feats=64, 
                 norm_type=None, 
                 act_type='leakyrelu', 
                 use_channel_attention=True,
                 res_scale=1
                 ):
        super(ResBlock, self).__init__()

        blocks = []
        for i in range(2):
            blocks.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))
            if norm_type:
                blocks.append(get_norm(norm_type, n_feats))
            if act_type and i==0:
                blocks.append(get_act(act_type))        
        if use_channel_attention:
            blocks.append(SEModule(n_feats))
        self.blocks = nn.Sequential(*blocks) 
        self.res_scale = res_scale

    def forward(self, x):
        res = self.blocks(x)
        res = res * self.res_scale
        output = res + x
        return output 


class simpleResNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n_feats, 
                 n_blocks, 
                 norm_type=None, 
                 act_type='leakyrelu', 
                 use_channel_attention=True, 
                 use_global_residual=True, 
                 use_tanh=True,
                 res_scale=1):
        super(simpleResNet, self).__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1, bias=True)
        
        body = [ResBlock(n_feats, norm_type, act_type, use_channel_attention, res_scale=res_scale) for _ in range(n_blocks)]        
        self.body = nn.Sequential(*body)

        self.tail = nn.Conv2d(n_feats, out_channels, 3, 1, 1)

        self.use_global_residual = use_global_residual
        self.use_tanh = use_tanh
        
    def forward(self, x):       
        x = self.head(x)
        output = self.body(x)        
        if self.use_global_residual:
            output = output + x
        output = self.tail(output)
        if self.use_tanh:
            output = torch.tanh(output)        
        return output


class simpleUshapeNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n_feats=64, 
                 n_blocks=16, 
                 norm_type=None, 
                 act_type='leakyrelu',
                 use_channel_attention=True,
                 use_global_residual=True, 
                 use_tanh=True,
                 res_scale=1):
        super(simpleUshapeNet, self).__init__()
        self.use_global_residual = use_global_residual
        # self.use_tanh = use_tanh

        # self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)

        self.en1 = nn.Sequential(            
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
        )

        self.down1 = nn.Conv2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1)        
        self.en2 = nn.Sequential(            
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
        )

        self.down2 = nn.Conv2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1)

        blocks = []
        for _ in range(n_blocks):
            block = ResBlock(n_feats, norm_type, act_type, use_channel_attention, res_scale=res_scale)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.up1 = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1)

        self.de1 = nn.Sequential(            
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
        )

        self.up2 = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1)

        self.de2 = nn.Sequential(            
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
            ResBlock(n_feats, norm_type, act_type, res_scale=res_scale),
        )


    def forward(self, x):        
        # x = self.head(x)
        e1 = self.en1(x)
        x = self.down1(e1)
        
        e2 = self.en2(x)
        res = self.down2(e2)
        
        x = self.middle(res)
        if self.use_global_residual:
            x = x + res
        
        
        x = self.up1(x)
        if self.use_global_residual:
            x = x + e2
        x = self.de1(x)
        
        x = self.up2(x)
        if self.use_global_residual:
            x = x + e1
        x = self.de2(x)        

        return x       


class MSPL_Generator(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 n_feats=128, 
                 n_blocks=[4,4,4,4], 
                 norm_type=None, 
                 act_type='leakyrelu',
                 use_channel_attention=True,
                 use_global_residual=True, 
                 use_tanh=False):
        super(MSPL_Generator, self).__init__()
        
        self.from_rgb = nn.Conv2d(in_channels, n_feats, (1, 1), bias=True) 

        self.net0 = simpleUshapeNet(n_feats, n_feats, n_feats, n_blocks[0], norm_type, act_type, use_channel_attention, use_global_residual, use_tanh)
        self.to_rgb0 = nn.Conv2d(n_feats, out_channels, (1, 1), bias=True) 
        
        self.net1 = simpleUshapeNet(n_feats, n_feats, n_feats, n_blocks[1], norm_type, act_type, use_channel_attention, use_global_residual, use_tanh)
        self.to_rgb1 = nn.Conv2d(n_feats, out_channels, (1, 1), bias=True) 

        self.net2 = simpleUshapeNet(n_feats, n_feats, n_feats, n_blocks[2], norm_type, act_type, use_channel_attention, use_global_residual, use_tanh)
        self.to_rgb2 = nn.Conv2d(n_feats, out_channels, (1, 1), bias=True) 
        
        self.net3 = simpleUshapeNet(n_feats, n_feats, n_feats, n_blocks[3], norm_type, act_type, use_channel_attention, use_global_residual, use_tanh)        
        self.to_rgb3 = nn.Conv2d(n_feats, out_channels, (1, 1), bias=True)

    def forward(self, x):  
        
        feat = self.from_rgb(x) 

        feat0 = self.net0(feat)
        feat1 = self.net1(feat0)
        feat2 = self.net2(feat1)
        feat3 = self.net3(feat2) 
        # feat4 = self.net4(feat3)

        out0 = torch.tanh(self.to_rgb0(feat0)) 
        out1 = torch.tanh(self.to_rgb1(feat1)) 
        out2 = torch.tanh(self.to_rgb2(feat2)) 
        out3 = torch.tanh(self.to_rgb3(feat3))
        # out4 = torch.tanh(self.to_rgb3(feat4))   
        return out0, out1, out2, out3

