import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import numpy as np


class RGB_net(nn.Module):
    def __init__(self):
        super(RGB_net, self).__init__()
        net = torchvision.models.resnet18()
        features_rgb = list(net.children())
        self.net = nn.Sequential(*features_rgb[0:8])
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)
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
        gap = self.gavg_pool(y).squeeze()
        gap = func.sigmoid(gap)
        # print('[RGB-backbone]\ty_rgb: {}\tgap_rgb: {}'.format(y.size(), gap.size()))
        p = self.classifier(gap)
        return gap, p


class Depth_net(nn.Module):
    def __init__(self):
        super(Depth_net, self).__init__()
        net = torchvision.models.resnet18()
        features_d = list(net.children())
        features_d[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net = nn.Sequential(*features_d[0:8])
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)
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
        gap = self.gavg_pool(y).squeeze()
        gap = func.sigmoid(gap)
        #print('[D-backbone]\ty_d: {}\tgap_d: {}'.format(y.size(), gap.size()))
        q = self.classifier(gap)
        return gap, q
