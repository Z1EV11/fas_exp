import torch
import torch.nn as nn


"""
SimAM: Simple, Parameter-Free Attention Module
"""
class Sim_AM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(Sim_AM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


"""
NAM: Normalization-based Attention Module
"""
class NAM_Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(NAM_Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual #
        
        return x


class NAM(nn.Module):
    def __init__(self, channels,shape, out_channels=None, no_spatial=True):
        super(NAM, self).__init__()
        self.Channel_Att = NAM_Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
 
        return x_out1 


"""
Global Attention Mechanism
"""
class GAM_Attention(nn.Module):  
    def __init__(self, in_channels, out_channels, rate=4):  
        super(GAM_Attention, self).__init__()  

        self.channel_attention = nn.Sequential(  
            nn.Linear(in_channels, int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Linear(int(in_channels / rate), in_channels)  
        )  
      
        self.spatial_attention = nn.Sequential(  
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  
            nn.BatchNorm2d(int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),  
            nn.BatchNorm2d(out_channels)  
        )  
      
    def forward(self, x):  
        b, c, h, w = x.shape  
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)  
      
        x = x * x_channel_att  
      
        x_spatial_att = self.spatial_attention(x).sigmoid()  
        out = x * x_spatial_att  
      
        return out  