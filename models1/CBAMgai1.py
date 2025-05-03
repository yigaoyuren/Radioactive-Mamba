import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from models1.DCNV3 import *
from models1.memory_1 import MemoryMatrixBlock1
class SpatialAttention_max(nn.Module):
    def __init__(self, in_channels, reduction1=16, reduction2=8):
        super(SpatialAttention_max, self).__init__()
        self.inc = torch.tensor(in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_spatial = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction1, in_channels, bias=False),
        )

        self.fc_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction2, in_channels, bias=False),
        )

        self._init_weight()

    def forward(self, x):

        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        # print(y_avg.shape,"y_avg.shape")

        y_spatial = self.fc_spatial(y_avg).view(b, c, 1, 1)
        y_channel = self.fc_channel(y_avg).view(b, c, 1, 1)
        y_channel = y_channel.sigmoid()
        y_spatial=y_spatial.sigmoid()

        # map = (x * (y_spatial)).sum(dim=1) / self.inc
        # map = (map / self.inc).sigmoid().unsqueeze(dim=1)
        return y_spatial, y_channel

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)



class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())
        self.conv2d=nn.Conv2d(dim,dim*2,kernel_size=1)
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        # x=self.conv2d(x1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class CBAM(nn.Module):
    def __init__(self,in_channels):
        super(CBAM, self).__init__()
        self.SpatialAttention_max=SpatialAttention_max(in_channels)
        self.SpatialWeights=SpatialWeights(in_channels)
        self.conv2d=nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=1)
        self.DCNv2=DCNv2(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
        self.MemoryMatrixBlock1=MemoryMatrixBlock1(32,512,4)
    def forward(self,x):
        x1,x2=self.SpatialAttention_max(x)
        # print(x1.shape,x2.shape)
        # x3=self.conv2d(x)
        x3=self.DCNv2(x)
        x4=self.SpatialWeights(x,x3)
        # print(x4[0].shape)
        x1=x1*x4[0]
        x2=x2*x4[1]
        x=x+x1+x2
        x=self.MemoryMatrixBlock1(x)
        return x
if __name__ == '__main__':
    fre = torch.randn(64,512, 4, 4)
    rgb = torch.randn(64, 512, 4, 4)
    fff = CBAM(512)

    fre_mix = fff(rgb)
    print("fre_mix", fre_mix.shape)