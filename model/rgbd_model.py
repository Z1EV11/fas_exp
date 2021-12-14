import math

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torchvision import models


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
            p: probability of real in rgb branch
            q: probability of real in depth branch
            r: probability of real in joint branch
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

        return gap, r, p, q


class RGB_net(nn.Module):
    def __init__(self):
        super(RGB_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

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



class RGBDMH(nn.Module):

    """ 

    Two-stream RGBD architecture (old version)

    Attributes
    ----------
    pretrained: bool
        If set to `True` uses the pretrained DenseNet model as the base. If set to `False`, the network
        will be trained from scratch. 
        default: True 
    num_channels: int
        Number of channels in the input.      
    """

    def __init__(self, pretrained=True, num_channels=4):

        """ Init function

        Parameters
        ----------
        pretrained: bool
            If set to `True` uses the pretrained densenet model as the base. Else, it uses the default network
            default: True
        num_channels: int
            Number of channels in the input. 
        """
        super(RGBDMH, self).__init__()
        
        dense_rgb = models.densenet161(pretrained=pretrained) # import densenet

        dense_d = models.densenet161(pretrained=pretrained)
        
        features_rgb = list(dense_rgb.features.children()) # densenet's feature

        features_d = list(dense_d.features.children())

        temp_layer = features_d[0] # 

        mean_weight = np.mean(temp_layer.weight.data.detach().numpy(),axis=1) # for 96 filters

        new_weight = np.zeros((96,1,7,7))
  
        for i in range(1):
            new_weight[:,i,:,:]=mean_weight

        features_d[0]=nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        features_d[0].weight.data = torch.Tensor(new_weight)

        self.enc_rgb = nn.Sequential(*features_rgb[0:8])

        self.enc_d = nn.Sequential(*features_d[0:8])

        self.linear=nn.Linear(768,1)

        self.linear_rgb=nn.Linear(384,1)

        self.linear_d=nn.Linear(384,1)

        self.gavg_pool=nn.AdaptiveAvgPool2d(1)


    def forward(self, img_rgb, img_d):
        """ Propagate data through the network

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 
          The data to forward through the network. Expects Multi-channel images of size num_channelsx224x224

        Returns
        -------
        dec: :py:class:`torch.Tensor` 
            Binary map of size 1x14x14
        op: :py:class:`torch.Tensor`
            Final binary score.  
        gap: Gobal averaged pooling from the encoded feature maps

        """

        # x_rgb = img[:, [0,1,2], :, :]

        # x_depth = img[:, 3, :, :].unsqueeze(1)
        x_rgb = img_rgb
        x_depth = img_d[:, 0:1, :, :]

        enc_rgb = self.enc_rgb(x_rgb)

        enc_d = self.enc_d(x_depth)


        gap_rgb = self.gavg_pool(enc_rgb).squeeze() 
        gap_d = self.gavg_pool(enc_d).squeeze() 

        gap_d=gap_d.view(-1,384)

        gap_rgb=gap_rgb.view(-1,384)
        
        gap_rgb = nn.Sigmoid()(gap_rgb) 
        gap_d = nn.Sigmoid()(gap_d) 

        op_rgb=self.linear_rgb(gap_rgb)

        op_d=self.linear_d(gap_d)


        op_rgb = nn.Sigmoid()(op_rgb)   # p

        op_d = nn.Sigmoid()(op_d)   # q

        gap=torch.cat([gap_rgb,gap_d], dim=1)

        op = self.linear(gap)

        op = nn.Sigmoid()(op)   # r
 
        return gap, op, op_rgb, op_d
