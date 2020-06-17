import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module
    

class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(SimpleDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        outputs = self.conv5(conv4)

        if self.use_sigmoid:
            outputs = torch.sigmoid(outputs)
            
        return outputs


class MSPDiscriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(MSPDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        self.conv1_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, imgs):
        img1, img2, img3, img4 = imgs
        
        x1 = self.conv1_head(img1)
        conv1 = self.conv1(x1)

        x2 = self.conv2_head(img2)
        x2 = torch.cat((x2, conv1), dim=1)
        conv2 = self.conv2(x2)

        x3 = self.conv3_head(img3)
        x3 = torch.cat((x3, conv2), dim=1)
        conv3 = self.conv3(x3)

        x4 = self.conv4_head(img4)
        x4 = torch.cat((x4, conv3), dim=1)
        conv4 = self.conv4(x4)
        outputs = self.conv5(conv4)

        if self.use_sigmoid:
            outputs = torch.sigmoid(outputs)
            
        return outputs


class MSPDiscriminator_01(nn.Module):
    
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(MSPDiscriminator_01, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        self.conv1_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5_head = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv6 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, imgs):
        img1, img2, img3, img4, img5 = imgs
        
        x1 = self.conv1_head(img1)
        conv1 = self.conv1(x1)

        x2 = self.conv2_head(img2)
        x2 = torch.cat((x2, conv1), dim=1)
        conv2 = self.conv2(x2)

        x3 = self.conv3_head(img3)
        x3 = torch.cat((x3, conv2), dim=1)
        conv3 = self.conv3(x3)

        x4 = self.conv4_head(img4)
        x4 = torch.cat((x4, conv3), dim=1)
        conv4 = self.conv4(x4)

        x5 = self.conv5_head(img5)
        x5 = torch.cat((x5, conv4), dim=1)
        conv5 = self.conv5(x5)
        outputs = self.conv6(conv5)

        if self.use_sigmoid:
            outputs = torch.sigmoid(outputs)
            
        return outputs