import torch
import torch.nn as nn


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


class Dense_net(nn.Module):
    def __init__(self):
        super(Dense_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        net.add_module("BN", nn.BatchNorm2d(num_channels))
        net.add_module("relu", nn.ReLU())
        net.add_module("global_avg_pool", Global_avg_pool2d()) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
        net.add_module("fc", nn.Sequential(Flatten_layer(), nn.Linear(num_channels, 10))) 

    def forward(self, x):
        y = self.net(x)
        return y