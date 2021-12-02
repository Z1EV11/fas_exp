import torch
import torch.nn as nn


# common
def cd_conv2d(in_channels, out_channels):
    return 0

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels), 
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


# model
class rgbd_model(nn.Module):
    def __init__(self, rgb_model, d_model, mode):
        super(rgbd_model, self).__init__()
        self.rgb_model = rgbd_model
        self.d_model = d_model

    def forward(self, x):
        y_rgb = rgb_model(x)
        y_d = d_model(x)   

    def backward(self):
        pass


class rgb_model(nn.Module):
    def __init__(self):
        super(rgb_model, self).__init__()
        self.rgb_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        self.rgb_net(x)


class d_model(nn.Module):
    def __init__(self):
        super(d_model, self).__init__()
        self.d_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def forward(self, x):
        self.d_net(x)