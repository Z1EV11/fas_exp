import math

import torch
import torch.nn as nn
import torch.nn.functional as func

from .res_net import RGB_net, Depth_net


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
