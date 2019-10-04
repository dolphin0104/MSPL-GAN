from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#================================================================================
# Helpers
#================================================================================
def get_act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation [{:s}] is not found'.format(act_type))
    return layer


def get_norm(norm_type, nc, affine=True):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=affine)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=affine)
    else:
        raise NotImplementedError(
            'normalization [{:s}] is not found'.format(norm_type))
    return layer


#================================================================================
# Architecture useful blocks
#================================================================================
class BasicConv(nn.Module):
    """conv-bn-act"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
        stride=1, padding=0, dilation=1, groups=1, 
        act_type='relu', norm_type='batch', bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            dilation=dilation, groups=groups, bias=bias)        
        self.bn = get_norm(norm_type, out_channels) if norm_type else None
        self.act = get_act(act_type) if act_type else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self, norm_type, act_type):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, 
            stride=1, padding=(kernel_size-1) // 2, 
            norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, norm_type, act_type,
         reduction_ratio=16, pool_types=['avg', 'max'], use_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.use_spatial = use_spatial
        if use_spatial:
            self.SpatialGate = SpatialGate(norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if self.use_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class ResNetBlock(nn.Module):
    def __init__(self, n_feat, kernel_size=3, stride=1, dilation=1, groups=1,
        bias=True, norm_type=None, act_type='leakyrelu', res_scale=1, use_channel=False, use_spatial=False):
        super(ResNetBlock, self).__init__()
        self.conv0 = BasicConv(n_feat, n_feat, kernel_size=kernel_size, 
            stride=stride, dilation=dilation, groups=groups, bias=bias, 
            norm_type=None, act_type=None)
       
        self.conv1 = BasicConv(n_feat, n_feat, kernel_size=kernel_size, 
            stride=stride, dilation=dilation, groups=groups, bias=bias, 
            norm_type=None, act_type=None)

        self.res_scale = res_scale
        self.attention_module = None
        if use_channel:
            self.attention_module = CBAM(
                n_feat, norm_type=norm_type, act_type=act_type,
                use_spatial=use_spatial)
                   
    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.conv1(out)
        if self.attention_module:
            out = self.attention_module(out)
        out = out.mul(self.res_scale)
        out += residual
        return out


