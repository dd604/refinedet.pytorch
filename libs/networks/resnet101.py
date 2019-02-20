# -*- coding: utf-8 -*-
# --------------------------------------------------------
# RefineDet in PyTorch 
# Written by Dongdong Wang
# Official and original Caffe implementation is at
# https://github.com/sfzhang15/RefineDet
# --------------------------------------------------------

import os
import torch
import torch.nn as nn

from torchvision.models.resnet import Bottleneck, resnet101
# from resnet import Bottleneck, resnet101
import pdb

is_batchnorm = True
layers_out_channels = [512, 1024, 2048, 512]


class ExtraResModule(nn.Module):
    """
    """
    def __init__(self, in_channels, internal_channels):
        """
        :param in_channels: number of forward feature channles
        :param internal_channels: number of internal channels
        """
        super(ExtraResModule, self).__init__()
        stride = 2
        expansion = 4
        # pdb.set_trace()
        out_channels = expansion * internal_channels
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.resbody = Bottleneck(in_channels, internal_channels,
                                  stride=stride, downsample=downsample)
        # self.downsample = downsample
    
    def forward(self, x):
        return self.resbody(x)


