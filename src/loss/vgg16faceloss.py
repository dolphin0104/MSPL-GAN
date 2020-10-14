import  torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as py_models
import numpy as np


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        checkpoint = torch.load(model_path)
        vgg16 = py_models.vgg16(num_classes=2622)
        vgg16.load_state_dict(checkpoint['state_dict'])
        self.features = nn.Sequential(*vgg16.features[:13])
        # self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        # self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        # fix the encoder
        # for i in range(3):
        #     for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
        #         param.requires_grad = False
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, image):
        x = self.features(image)
        return x



# class vgg16FaceLoss(nn.Module):
#     def __init__(self, model_path, p_layer=14, device=torch.device('cpu')):
#         super(vgg16FaceLoss, self).__init__()
#         self.criterion = nn.L1Loss()
# #         conv_3_3_layer = 14
#         checkpoint = torch.load(model_path)
#         vgg16 = py_models.vgg16(num_classes=2622)
#         vgg16.load_state_dict(checkpoint['state_dict'])
#         cnn = vgg16.features
#         cnn = cnn.to(device)
#         model = nn.Sequential()
#         model = model.to(device)
#         for i,layer in enumerate(list(cnn)):
#             model.add_module(str(i),layer)
#             if i == p_layer:
#                 break
#         self.contentFunc = model   
#         del vgg16, cnn, checkpoint


#     def getloss(self, fakeIm, realIm):
#         # if isinstance(fakeIm, numpy.ndarray):
#         #     fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
#         #     realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
#         f_fake = self.contentFunc.forward(fakeIm)
#         f_real = self.contentFunc.forward(realIm)
#         f_real_no_grad = f_real.detach()
#         loss = self.criterion(f_fake, f_real_no_grad)
#         return loss