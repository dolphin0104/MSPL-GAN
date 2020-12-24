import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.models as py_models

class PerceptualLoss():
    def __init__(self, loss, gpu=0, p_layer=14):
        super(PerceptualLoss, self).__init__()
        self.criterion = loss
        
        cnn = py_models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == p_layer:
                break
        self.contentFunc = model     

    def getloss(self, fakeIm, realIm):
        if isinstance(fakeIm, numpy.ndarray):
            fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
            realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
class PerceptualLoss16():
    def __init__(self, loss, gpu=0, p_layer=14):
        super(PerceptualLoss16, self).__init__()
        self.criterion = loss
#         conv_3_3_layer = 14
        checkpoint = torch.load('D:/Github/03.weight/cvpr2019/checkpoint/vggface/VGGFace16.pth')
        vgg16 = py_models.vgg16(num_classes=2622)
        vgg16.load_state_dict(checkpoint['state_dict'])
        cnn = vgg16.features
        cnn = cnn.cuda()
#         cnn = cnn.to(gpu)
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
#             print(layer)
            model.add_module(str(i),layer)
            if i == p_layer:
                break
        self.contentFunc = model   
        del vgg16, cnn, checkpoint

    def getloss(self, fakeIm, realIm):
        if isinstance(fakeIm, numpy.ndarray):
            fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
            realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss