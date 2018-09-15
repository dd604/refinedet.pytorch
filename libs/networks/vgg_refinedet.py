# encoding: utf-8

import os
import torch
import torch.nn as nn


from vgg import make_vgg_layers, add_extra_layers, \
    key_layer_ids, layers_out_channels
from refinedet import RefineDet as _RefineDet
from libs.utils.net_utils import L2Norm
import pdb

class VGGRefineDet(_RefineDet):
    """
    """
    def __init__(self, num_classes, cfg):
        super(VGGRefineDet, self).__init__(num_classes, cfg)
        # _RefineDet.__init__(self, num_classes, cfg)
        
    
    def _init_modules(self, model_path=None, pretrained=True):
        self.base = nn.ModuleList(make_vgg_layers())
        self.extra = nn.ModuleList(add_extra_layers())
        # pdb.set_trace()
        self.pretrained = pretrained
        self.model_path = model_path
        if self.pretrained == True and model_path is not None:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            self.base.load_state_dict({k: v for k, v in state_dict.items()
                                       if k in self.base.state_dict()})
            
            # fix weights
            for param in self.base.parameters():
                # if not (param in state_dict.items()):
                #     continue
                param.requires_grad = False

        self.layers_out_channels = layers_out_channels
    
        # construct base network
        assert key_layer_ids[2] == -1 and key_layer_ids[3] == -1, \
            'Must use outputs of the final layers in base and extra.'
        # pdb.set_trace()
        base_layers = list(self.base.children())
        self.layer1 = nn.ModuleList(base_layers[:key_layer_ids[0]])
        self.layer2 = nn.ModuleList(base_layers[key_layer_ids[0] : key_layer_ids[1]])
        self.layer3 = nn.ModuleList(base_layers[key_layer_ids[1]:])
        self.layer4 = nn.ModuleList(list(self.extra.children()))
        # L2Norm has been initialized while building.
        # self.L2Norm_conv4_3 = L2Norm(self.layers_out_channels[0], 20)
        # self.L2Norm_conv5_3 = L2Norm(self.layers_out_channels[1], 16)
        self.L2Norm_conv4_3 = L2Norm(self.layers_out_channels[0], 10)
        self.L2Norm_conv5_3 = L2Norm(self.layers_out_channels[1], 8)

        # build pyramid layers and other parts
        super(VGGRefineDet, self)._init_part_modules()
    
    def _get_forward_features(self, x):
        """
        Calculate forward_features = [c1, c2, c3, c4]
        :param x:
        :return:
        """
        forward_features = []
        # c1
        for k in xrange(len(self.layer1)):
            x = self.layer1[k](x)
        x = self.L2Norm_conv4_3(x)
        forward_features.append(x)
        # c2
        for k in xrange(len(self.layer2)):
            x = self.layer2[k](x)
        x = self.L2Norm_conv5_3(x)
        forward_features.append(x)
        # c3
        for k in xrange(len(self.layer3)):
            x = self.layer3[k](x)
        forward_features.append(x)
        # c4
        for k in xrange(len(self.layer4)):
            x = self.layer4[k](x)
        forward_features.append(x)
        
        return forward_features
    
