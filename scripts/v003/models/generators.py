from collections import OrderedDict
import torch
import torch.nn as nn
from models import blocks as B


class single_Net(nn.Module):
    def __init__(self, n_blocks, in_channels, out_channels, n_feat=64,
        norm_type=None, act_type='leakyrelu',
        res_scale=1, use_channel=False, use_spatial=False):
        super(single_Net, self).__init__()
        
        self.head = B.BasicConv(in_channels, n_feat, norm_type=norm_type, act_type=act_type)
                
        body = [B.ResNetBlock(n_feat=n_feat, norm_type=norm_type, act_type=act_type,
            res_scale=res_scale, use_channel=use_channel, use_spatial=use_spatial) 
            for _ in range(n_blocks)]
        self.body = nn.Sequential(*body)

        self.tail0 = B.BasicConv(n_feat, n_feat, norm_type=norm_type, act_type=act_type)
        self.tail1 = B.BasicConv(n_feat, out_channels, norm_type=norm_type, act_type=act_type)
    
    def forward(self, x):
        residual_feat = self.head(x)
        out = self.body(residual_feat)
        out += residual_feat
        out = self.tail0(out)
        out = self.tail1(out)
        return out


#TODO
# class multi_serial_Net(nn.Module):
#     def __init__(self, n_blocks, in_channels, out_channels, n_feat=64):
#         super(multi_serial_Net, self).__init__()
        
#         self.head = B.BasicConv(in_channels, n_feat, norm_type=None, act_type='leakyrelu')
                
#         body = [B.ResNetBlock(n_feat=n_feat) for _ in range(n_blocks)]
#         self.body = nn.Sequential(*body)

#         self.tail0 = B.BasicConv(n_feat, n_feat, norm_type=None, act_type='leakyrelu')
#         self.tail1 = B.BasicConv(n_feat, out_channels, norm_type=None, act_type='leakyrelu')
    
#     def forward(self, x):
#         residual_feat = self.head(x)
#         out = self.body(residual_feat)
#         out += residual_feat
#         out = self.tail0(out)
#         out = self.tail1(out)
#         return out       


# class multi_parallel_Net(nn.Module):
#     def __init__(self, n_blocks, in_channels, out_channels, n_feat=64):
#         super(multi_parallel_Net, self).__init__()
        
#         self.head = B.BasicConv(in_channels, n_feat, norm_type=None, act_type='leakyrelu')
                
#         body = [B.ResNetBlock(n_feat=n_feat) for _ in range(n_blocks)]
#         self.body = nn.Sequential(*body)

#         self.tail0 = B.BasicConv(n_feat, n_feat, norm_type=None, act_type='leakyrelu')
#         self.tail1 = B.BasicConv(n_feat, out_channels, norm_type=None, act_type='leakyrelu')
    
#     def forward(self, x):
#         residual_feat = self.head(x)
#         out = self.body(residual_feat)
#         out += residual_feat
#         out = self.tail0(out)
#         out = self.tail1(out)
#         return out 
