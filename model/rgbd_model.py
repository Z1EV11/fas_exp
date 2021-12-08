import math

import torch
import torch.nn as nn
import torch.nn.functional as func


""" Common """
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

# 全局平均池化层：可通过将池化窗口形状设置成输入的高和宽实现
class Global_avg_pool2d(nn.Module):
    def __init__(self):
        super(Global_avg_pool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class Flatten_layer(torch.nn.Module):
    def __init__(self):
        super(Flatten_layer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

# DenseNet
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk

class Dense_block(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(Dense_block, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X


""" Model """
# RGB-D PAD
class RGBD_model(nn.Module):
    def __init__(self, rgb_net, depth_net):
        super(RGBD_model, self).__init__()
        self.rgb_net = rgb_net()
        self.depth_net = depth_net()

    def forward(self, rgb_map, depth_map):
        p = self.rgb_net(rgb_map)
        q = self.depth_net(depth_map)
        r = 1
        return p, q, r


# DenseNet
class RGB_net(nn.Module):
    def __init__(self):
        super(RGB_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        net.add_module("BN", nn.BatchNorm2d(num_channels))
        net.add_module("relu", nn.ReLU())
        net.add_module("global_avg_pool", Global_avg_pool2d()) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
        net.add_module("fc", nn.Sequential(Flatten_layer(), nn.Linear(num_channels, 10))) 

    def forward(self, x):
        y = self.net(x)
        return y


# DenseNet
class Depth_net(nn.Module):
    def __init__(self):
        super(Depth_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def forward(self, x):
        y = self.net(x)
        return y