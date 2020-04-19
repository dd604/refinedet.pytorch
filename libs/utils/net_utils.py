# -*- coding: utf-8 -*-
# --------------------------------------------------------
# RefineDet in PyTorch
# Written by Dongdong Wang
# Official and original Caffe implementation is at
# https://github.com/sfzhang15/RefineDet
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as init
import pdb


def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    # print(m)
    if isinstance(m, nn.Conv1d):
        init.normal(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    # else:
    #     print('Warning, an unknowned instance!!')
    #     print(m)

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        # pdb.set_trace()
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class TCB(nn.Module):
    """
    Transfer Connection Block Architecture
    This block
    """
    def __init__(self, lateral_channels, channles,
                 internal_channels=256, is_batchnorm=False):
        """
        :param lateral_channels: number of forward feature channles
        :param channles: number of pyramid feature channles
        :param internal_channels: number of internal channels
        """
        super(TCB, self).__init__()
        self.is_batchnorm = is_batchnorm
        # Use bias if is_batchnorm is False, donot otherwise.
        use_bias = not self.is_batchnorm
        # conv + bn + relu
        self.conv1 = nn.Conv2d(lateral_channels, internal_channels,
                               kernel_size=3, padding=1, bias=use_bias)
        # ((conv2 + bn2) element-wise add  (deconv + deconv_bn)) + relu
        # batch normalization before element-wise addition
        self.conv2 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1, bias=use_bias)
        self.deconv = nn.ConvTranspose2d(channles, internal_channels,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)
        # conv + bn + relu
        self.conv3 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1, bias=use_bias)
        self.relu = nn.ReLU(inplace=True)
        
        if self.is_batchnorm:
            self.bn1 = nn.BatchNorm2d(internal_channels)
            self.bn2 = nn.BatchNorm2d(internal_channels)
            self.deconv_bn = nn.BatchNorm2d(internal_channels)
            self.bn3 = nn.BatchNorm2d(internal_channels)
        # attribution
        self.out_channels = internal_channels
    
    def forward(self, lateral, x):
        if self.is_batchnorm:
            lateral_out = self.relu(self.bn1(self.conv1(lateral)))
            # element-wise addation
            out = self.relu(self.bn2(self.conv2(lateral_out)) +
                            self.deconv_bn(self.deconv(x)))
            out = self.relu(self.bn3(self.conv3(out)))
        else:
            # no batchnorm
            lateral_out = self.relu(self.conv1(lateral))
            # element-wise addation
            out = self.relu(self.conv2(lateral_out) + self.deconv(x))
            out = self.relu(self.conv3(out))
        
        return out


def make_special_tcb_layer(in_channels, internal_channels,
                           is_batchnorm=False):
    # layers = list()
    if is_batchnorm:
        layers = [nn.Conv2d(in_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.BatchNorm2d(internal_channels),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(internal_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.BatchNorm2d(internal_channels),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(internal_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.BatchNorm2d(internal_channels),
                  nn.ReLU(inplace=True)]

    else:
        layers = [nn.Conv2d(in_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(internal_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(internal_channels, internal_channels,
                            kernel_size=3, padding=1),
                  nn.ReLU(inplace=True)]
    return layers