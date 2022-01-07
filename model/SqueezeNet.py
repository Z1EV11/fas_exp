import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import numpy as np


class RGB_net(nn.Module):
    def __init__(self):
        super(RGB_net, self).__init__()
        net = torchvision.models.squeezenet1_1()
        features_rgb = list(net.features.children())
        self.net = nn.Sequential(*features_rgb)
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
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
        net = torchvision.models.squeezenet1_1()
        features_d = list(net.features.children())
        temp_layer = features_d[0]
        mean_weight = np.mean(temp_layer.weight.data.detach().numpy(),axis=1) # for 96 filters
        new_weight = np.zeros((64,1,3,3))
        for i in range(1):
            new_weight[:,i,:,:]=mean_weight
        features_d[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
        features_d[0].weight.data = torch.Tensor(new_weight)
        self.net = nn.Sequential(*features_d)
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
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