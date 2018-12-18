# encoding: utf-8

import os
import torch
import torch.nn as nn


from resnet101 import resnet101, ExtraResModule, is_batchnorm, \
    layers_out_channels
from refinedet import RefineDet as _RefineDet

from libs.utils.net_utils import L2Norm
import pdb

class ResNetRefineDet(_RefineDet):
    """
    """
    def __init__(self, num_classes, cfg):
        super(ResNetRefineDet, self).__init__(num_classes, cfg)
    
    def _init_modules(self, model_path=None, pretrained=True,
                      fine_tuning=True):
        self.base = resnet101(pretrained=False)
        self.pretrained = pretrained
        self.model_path = model_path
        # self.fine_tuning = self.fine_tuning
        if self.pretrained == True and model_path is not None:
            print("Loading pretrained weights from %s" % self.model_path)
            state_dict = torch.load(self.model_path)
            self.base.load_state_dict({k: v for k, v in state_dict.items()
                                       if k in self.base.state_dict()})
            # fix weights
            if not fine_tuning:
                for param in self.base.parameters():
                    param.requires_grad = False
        
        self.layers_out_channels = layers_out_channels
        
        self.extra = ExtraResModule(self.layers_out_channels[-2],
                                    self.layers_out_channels[-1] // 4)
        self.layer1 = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu,
                                    self.base.maxpool, self.base.layer1,
                                    self.base.layer2)
        self.layer2 = self.base.layer3
        self.layer3 = self.base.layer4
        self.layer4 = self.extra
        # build pyramid layers and other parts
        self.is_batchnorm = True
        self._init_part_modules()
    
    def _get_forward_features(self, x):
        """
        Calculate forward_features = [c1, c2, c3, c4]
        :param x:
        :return:
        """
        forward_features = []
        # c1
        x = self.layer1(x)
        forward_features.append(x)
        # c2
        x = self.layer2(x)
        forward_features.append(x)
        # c3
        x = self.layer3(x)
        forward_features.append(x)
        # c4
        x = self.layer4(x)
        forward_features.append(x)
        
        return forward_features

