import math

import torch
import torch.nn as nn
import torch.nn.functional as func


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
        return func.avg_pool2d(x, kernel_size=x.size()[2:])

class Flatten_layer(torch.nn.Module):
    def __init__(self):
        super(Flatten_layer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

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


class RGBD_model(nn.Module):
    """
    RGB-D Multi-Modal PAD Framework
    Args:
        rgb_net
        depth_net
    """
    def __init__(self, rgb_net, depth_net):
        super(RGBD_model, self).__init__()
        self.rgb_net = rgb_net()
        self.depth_net = depth_net()
        self.gavg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc_lalyer_p = nn.Linear(1000, 1)
        self.fc_lalyer_r = nn.Linear(2000, 1)

    def forward(self, rgb_map, depth_map):
        """"
        Returns:
            p
            q
            r
        """
        y_rgb = self.rgb_net(rgb_map)
        y_d = self.depth_net(depth_map)

        gap_rgb = self.gavg_pool(y_rgb).squeeze()
        gap_d = self.gavg_pool(y_d).squeeze() 

        gap_rgb = nn.Sigmoid()(y_rgb) 
        gap_d = nn.Sigmoid()(y_d) 

        op_rgb=self.fc_lalyer_p(gap_rgb)
        op_d=self.fc_lalyer_p(gap_d)

        p = nn.Sigmoid()(op_rgb)
        q = nn.Sigmoid()(op_d)

        gap=torch.cat([gap_rgb,gap_d], dim=1)
        op = self.fc_lalyer_r(gap)
        r = nn.Sigmoid()(op)

        return p, q, r


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