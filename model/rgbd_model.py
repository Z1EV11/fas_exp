import math

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import torchvision

# import res_net, mobile_net


class CD_Conv2d(nn.Module):
    """
    Central Difference Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(CD_Conv2d, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = func.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff


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
            nn.Linear(1024,1),
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
        x_d = x_d[:,0:1,:,:] # [B,1,224,224]
        print('[RGBD-backbone]\tx_rgb: {}\tx_d: {}'.format(x_rgb.size(), x_d.size()))
        gap_rgb, p = self.rgb_net(x_rgb)   # [B,3,224,224] -> [B,384,14,14]
        print('[RGB-head]\tgap_rgb: {}\tp: {}'.format(gap_rgb.size(), p.size()))
        gap_d, q = self.depth_net(x_d) # [B,1,224,224] -> [B,384,14,14]
        print('[D-head]\tgap_d: {}\tq: {}'.format(gap_d.size(), q.size()))

        gap = torch.cat([gap_rgb,gap_d], dim=1) # [B, 4, 1, 1]
        r = self.classifier(gap)
        print('[RGBD-head]\t gap: {}\t r: {}'.format(gap.size(), r.size()))
        return gap, r, p, q


class RGB_net(nn.Module):
    def __init__(self):
        super(RGB_net, self).__init__()
        net = torchvision.models.resnet18()
        features_rgb = list(net.children())
        self.net = nn.Sequential(*features_rgb[0:8])
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Sigmoid(),
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
        print('[RGB-backbone]\ty_rgb: {}\tgap_rgb: {}'.format(y.size(), gap.size()))
        p = self.classifier(gap)
        return gap, p


class Depth_net(nn.Module):
    def __init__(self):
        super(Depth_net, self).__init__()
        net = torchvision.models.resnet18()
        features_d = list(net.children())
        # temp_layer = features_d[0]
        # mean_weight = np.mean(temp_layer.weight.data.detach().numpy(),axis=1) # for 96 filters
        # new_weight = np.zeros((64,1,7,7))
        # for i in range(1):
        #     new_weight[:,i,:,:]=mean_weight
        features_d[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # features_d[0].weight.data = torch.Tensor(new_weight)
        self.net = nn.Sequential(*features_d[0:8])
        self.gavg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Sigmoid(),
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
        print('[D-backbone]\ty_d: {}\tgap_d: {}'.format(y.size(), gap.size()))
        q = self.classifier(gap)
        return gap, q
