import math

import torch
import torch.nn as nn
import torch.nn.functional as func


"""
    Common
"""
# Cross Modal Focal Loss
class CMF_Loss(nn.Module):
    def __init__(self, alpha=1, gamma=1, lamb=0.7):
        super(CMF_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb

    def forward(self, p, q, r, label):
        ce_loss = nn.CrossEntropyLoss()
        total_loss = (1-self.lamb)*ce_loss(r) + self.lamb*(self.cmfl(p,q)+self.cmfl(q,p))
        return total_loss

    def w(p, q):
        return (2*p*(q^2))/(p+q)

    def cmfl(self, p, q):
        return -self.alpha*(1-self.w(p,q))^self.gamma*math.log(p)

# Central Difference Convolution
class CD_Conv2d(nn.Module):
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

# DenseNet - Dense Block
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

# DenseNet - Transition Block
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk


"""
    Model
"""
# RGB-D PAD
class rgbd_model(nn.Module):
    def __init__(self, rgb_model, d_model, mode):
        super(rgbd_model, self).__init__()
        self.rgb_model = rgb_model
        self.d_model = d_model

    def forward(self, rgb_map, d_map):
        p = rgb_model(rgb_map)
        q = d_model(d_map)
        r = 1
        return p, q, r


# DenseNet
class rgb_model(nn.Module):
    def __init__(self):
        super(rgb_model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        self.net(x)

# DenseNet
class d_model(nn.Module):
    def __init__(self):
        super(d_model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def forward(self, x):
        self.net(x)