import math
import functools
import numpy as np
import torch
import torch.nn as nn
from model.modules import basicblocks as B
from model.modules.model_utils import *

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, 
        norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator_112(nn.Module):
    def __init__(self, model_type='cRaGAN', base_nf=64, 
        norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_112, self).__init__()
        if model_type == 'cRaGAN': 
            in_nc = 6
        else: in_nc = 3
        self.model_type = model_type
        # features
        # hxw, c
        # input 112x96
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, 
            norm_type=None, act_type=act_type, mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, 
            stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, 
            stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, 
            stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, 
            stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, 
            stride=2, norm_type=norm_type, act_type=act_type, mode=mode)
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, 
            stride=1, norm_type=norm_type, act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, 
            stride=1, norm_type=norm_type, act_type=act_type, mode=mode) 
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, 
            stride=1, norm_type=norm_type, act_type=act_type, mode=mode)        
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, 
            conv5, conv6, conv7, conv8)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 126), nn.LeakyReLU(0.2, True), nn.Linear(126, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        