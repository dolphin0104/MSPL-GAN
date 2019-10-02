import math
import torch
import torch.nn as nn
import torchvision

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram  

class VGG16_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


# Assume input range is [0, 1]
class VGG19_FeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGG19_FeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

class VGG19_mulitFeatureExtractor(nn.Module):
    def __init__(self,
                 is_img=True,                 
                 #content_feature_layer=34, #19, #34 in EDSR code..
                 feature_layers=['0', '5', '10', '19', '28', '34'],
                 #['0', '5', '10', '19', '28'],  #Select conv1_1 ~ conv5_1 activation maps.
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGG19_mulitFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.select = feature_layers
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            if is_img:
                mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            else:
                mean = torch.Tensor([0.485-1, 0.456-1, 0.406-1]).view(1, 3, 1, 1).to(device)
                # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
                std = torch.Tensor([0.229*2, 0.224*2, 0.225*2]).view(1, 3, 1, 1).to(device)
                # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        # No need to BP to variable
        self.features = model.features
        for _, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output