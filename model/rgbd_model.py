import torch
import torch.nn as nn


<<<<<<< HEAD
""" Common """
# Cross Modal Focal Loss
class CMF_Loss(nn.Module):
    def __init__(self):
        super(CMF_Loss, self).__init__()
    def forward(self, x):
        pass

# Central Difference Conv
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
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

# DenseNet - Dense Block
=======
# common
def cd_conv2d(in_channels, out_channels):
    return 0

>>>>>>> 2eac43400ccff2b47f938d06efaff3b2feb29589
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels), 
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

<<<<<<< HEAD
# DenseNet - Transition Block
=======
>>>>>>> 2eac43400ccff2b47f938d06efaff3b2feb29589
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk


<<<<<<< HEAD
""" Model """
=======
# model
>>>>>>> 2eac43400ccff2b47f938d06efaff3b2feb29589
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
<<<<<<< HEAD
        self.net = nn.Sequential(
=======
        self.rgb_net = nn.Sequential(
>>>>>>> 2eac43400ccff2b47f938d06efaff3b2feb29589
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
<<<<<<< HEAD
        self.net(x)
=======
        self.rgb_net(x)
>>>>>>> 2eac43400ccff2b47f938d06efaff3b2feb29589


class d_model(nn.Module):
    def __init__(self):
        super(d_model, self).__init__()
<<<<<<< HEAD
        self.net = nn.Sequential(
=======
        self.d_net = nn.Sequential(
>>>>>>> 2eac43400ccff2b47f938d06efaff3b2feb29589
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def forward(self, x):
<<<<<<< HEAD
        self.net(x)
=======
        self.d_net(x)
>>>>>>> 2eac43400ccff2b47f938d06efaff3b2feb29589
