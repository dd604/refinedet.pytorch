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


from .vgg import make_vgg_layers, add_extra_layers, \
    key_layer_ids, layers_out_channels, is_batchnorm
from .refinedet import RefineDet as _RefineDet
from libs.utils.net_utils import L2Norm
import pdb

class VGGRefineDet(_RefineDet):
    """
    RefineDet with VGG16 as the base network.
    """
    def __init__(self, num_classes, cfg):
        super(VGGRefineDet, self).__init__(num_classes, cfg)
        
    def _init_modules(self, base_model_path=None, pretrained=True,
            fine_tuning=True):
        """
        Initialize modules, load weights and fix parameters for a base model
        if necessary.
        :param base_model_path: model path for a base network.
        :param pretrained: whether load a pretrained model or not.
        :param fine_tuning: whether fix parameters for a base model.
        :return:
        """
        self.base = nn.ModuleList(make_vgg_layers())
        # pdb.set_trace()
        self.pretrained = pretrained
        self.base_model_path = base_model_path
        # self.fine_tuning = self.fine_tuning
        if (base_model_path is not None) and (self.pretrained == True):
            print("Loading pretrained weights from %s" % self.base_model_path)
            state_dict = torch.load(self.base_model_path)
            self.base.load_state_dict({k: v for k, v in state_dict.items()
                                       if k in self.base.state_dict()})
            # fix weights
            if not fine_tuning:
                for param in self.base.parameters():
                    param.requires_grad = False
            
        self.layers_out_channels = layers_out_channels
        self.extra = nn.ModuleList(add_extra_layers())
        #pdb.set_trace()
        # construct base network
        assert key_layer_ids[2] == -1 and key_layer_ids[3] == -1, \
            'Must use outputs of the final layers in base and extra.'
        # pdb.set_trace()
        base_layers = list(self.base.children())
        self.layer1 = nn.Sequential(*(base_layers[:key_layer_ids[0]]))
        self.layer2 = nn.Sequential(*(base_layers[key_layer_ids[0] : key_layer_ids[1]]))
        self.layer3 = nn.Sequential(*(base_layers[key_layer_ids[1]:]))
        self.layer4 = nn.Sequential(*(self.extra.children()))
        # L2Norm has been initialized while building.
        # self.L2Norm_conv4_3 = L2Norm(self.layers_out_channels[0], 40)
        # self.L2Norm_conv5_3 = L2Norm(self.layers_out_channels[1], 32)
        self.L2Norm_conv4_3 = L2Norm(self.layers_out_channels[0], 10)
        self.L2Norm_conv5_3 = L2Norm(self.layers_out_channels[1], 8)

        # build pyramid layers and other parts
        # super(VGGRefineDet, self)._init_part_modules()
        self._init_part_modules()
        
    def _get_forward_features(self, x):
        """
        Calculate forward features
        :param x: input variable, the size is (batch_size, height, width,
        channel)
        :return forward_features:  a list [c1, c2, c3, c4]
        """
        forward_features = []
        # c1
        x = self.layer1(x)
        feature = self.L2Norm_conv4_3(x)
        forward_features.append(feature)
        # c2
        x = self.layer2(x)
        feature = self.L2Norm_conv5_3(x)
        forward_features.append(feature)
        # c3
        x = self.layer3(x)
        forward_features.append(x)
        # c4
        x = self.layer4(x)
        forward_features.append(x)
        
        return forward_features
    
