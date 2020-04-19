# -*- coding: utf-8 -*-
# --------------------------------------------------------
# RefineDet in PyTorch
# Written by Dongdong Wang
# Official and original Caffe implementation is at
# https://github.com/sfzhang15/RefineDet
# --------------------------------------------------------


import torch.nn as nn


###
# name_maps = {'conv1_1': 0, 'relu1_1': 1,
#              'conv1_2': 2, 'relu1_2': 3,
#              'pool1': 4,
#              'conv2_1': 5, 'relu2_1': 6,
#              'conv2_2': 7, 'relu2_2': 8,
#              'pool2': 9,
#              'conv3_1': 10, 'relu3_1': 11,
#              'conv3_2': 12, 'relu3_2': 13,
#              'conv3_3': 14, 'relu3_3': 15,
#              'pool3': 16,
#              'conv4_1': 17, 'relu4_1': 18,
#              'conv4_2': 19, 'relu4_2': 20,
#              'conv4_3': 21, 'relu4_3': 22,
#              'pool4': 23,
#              'conv5_1': 24, 'relu5_1': 25,
#              'conv5_2': 26, 'relu5_2': 27,
#              'conv5_3': 28, 'relu5_3': 29,
#              'pool5': 30,
#              }

# 1024
# conv_fc6 with dilation
# conv_fc7
base = [64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'M',
        512, 512, 512, 'M',
        512, 512, 512, 'M',
        1024, 1024]

# after conv_fc7
# conv6_1, conv6_2
extras = [256, 512]

# conv4_3
# conv5_3
# conv_fc7
# conv6_2
# indice starting from 0
# conv4_3, relu, 22
# conv5_3, relu, 29; pool5, 30
# conv_fc7, relu, 34
# extra, relu, -1
key_layer_ids = [22+1, 29+1, -1, -1]
# out_channels of conv4_3, conv5_3, conv_fc7, conv6_2
# [512, 512, 1024, 512]
layers_out_channels = [base[12], base[16], base[19], extras[-1]]
is_batchnorm = False


def make_vgg_layers(cfg=base, batch_norm=False):
    """
    :param cfg:
    :param batch_norm:
    :return:
    """
    layers = []
    in_channels = 3
    for v in cfg[:-2]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    # after pool5
    dilation = 6
    # dilation = 3
    kernel_size = 3
    padding = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) // 2
    # cfg[-3] == 'M'
    conv_fc6 = nn.Conv2d(cfg[-4], cfg[-2], kernel_size=kernel_size,
                         padding=padding, dilation=dilation)
    conv_fc7 = nn.Conv2d(cfg[-2], cfg[-1], kernel_size=1)
    # [31 - 34]
    layers += [conv_fc6, nn.ReLU(inplace=True), conv_fc7, nn.ReLU(inplace=True)]
    assert (len(layers) == 35)
    # pdb.set_trace()
    # print(layers)
    
    return layers


def add_extra_layers(in_channels=base[-1], cfg=extras, batch_norm=False):
    """
    :param in_channels: number of channels of input
    :param cfg:
    :param batch_norm:
    :return:
    """
    # Extra layers added to VGG for feature scaling
    layers = []
    # conv6_1 and conv6_2
    for k, v in enumerate(cfg):
        if k == 0:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=2, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    
    return layers


