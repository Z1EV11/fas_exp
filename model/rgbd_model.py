import math

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import numpy as np

# from .squeeze_net import RGB_net, Depth_net
from .attention import NAM


class RGBD_model(nn.Module):
    """
    RGB-D Architecture
    """
    def __init__(self, device):
        """
        Args:
            device
        """
        super(RGBD_model, self).__init__()
        self.rgb_net = RGB_net().to(device)
        self.depth_net = Depth_net().to(device)
        self.classifier = nn.Sequential(
            # nn.Sigmoid(),
            nn.Linear(1024,1), # diy by backbone
            nn.Sigmoid()
        )

    def forward(self, x_rgb, x_d):
        """"
        Args:
            x_rgb: rgb-image. [B,3,W,W]
            x_d: d-image. [B,3,W,W]
        Returns:
            p: prob of live in rgb
            q: prob of live in depth
            r: prob of live
        """
        x_d = x_d[:,0:1,:,:]
        #print('[RGBD-backbone]\tx_rgb: {}\tx_d: {}'.format(x_rgb.size(), x_d.size()))
        gap_rgb, p = self.rgb_net(x_rgb)
        #print('[RGB-head]\tgap_rgb: {}\tp: {}'.format(gap_rgb.size(), p.size()))
        gap_d, q = self.depth_net(x_d)
        #print('[D-head]\tgap_d: {}\tq: {}'.format(gap_d.size(), q.size()))

        gap = torch.cat([gap_rgb,gap_d], dim=1)
        r = self.classifier(gap)
        #print('[RGBD-head]\t gap: {}\t r: {}'.format(gap.size(), r.size()))
        return gap, r, p, q


# -------------------------------------------------
class RGB_net(nn.Module):
    def __init__(self):
        super(RGB_net, self).__init__()
        net = torchvision.models.resnet34(pretrained=True)
        features_rgb = list(net.children())
        self.net = nn.Sequential(*features_rgb[0:8])
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)
        self.att = NAM(512)
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            nn.Linear(512,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: rgb-image. [B,3,W,W]
        """
        y = self.net(x)
        # print('[RGB-backbone]\ty_rgb: {}'.format(y.size()))
        y = self.att(y)
        gap = self.gavg_pool(y).squeeze()
        gap = func.sigmoid(gap)
        # print('[RGB-backbone]\ty_rgb: {}\tgap_rgb: {}'.format(y.size(), gap.size()))
        p = self.classifier(gap)
        return gap, p


class Depth_net(nn.Module):
    def __init__(self):
        super(Depth_net, self).__init__()
        net = torchvision.models.resnet34(pretrained=True)
        features_d = list(net.children())
        temp_layer = features_d[0]
        mean_weight = np.mean(temp_layer.weight.data.detach().numpy(),axis=1) # for 96 filters
        new_weight = np.zeros((64,1,7,7))
        for i in range(1):
            new_weight[:,i,:,:]=mean_weight
        features_d[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        features_d[0].weight.data = torch.Tensor(new_weight)
        self.net = nn.Sequential(*features_d[0:8])
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)
        self.att = NAM(512)
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: d-image. [B,1,W,W]
        """
        y = self.net(x)
        # print('[D-backbone]\ty_d: {}'.format(y.size()))
        y = self.att(y)
        gap = self.gavg_pool(y).squeeze()
        gap = func.sigmoid(gap)
        #print('[D-backbone]\ty_d: {}\tgap_d: {}'.format(y.size(), gap.size()))
        q = self.classifier(gap)
        return gap, q