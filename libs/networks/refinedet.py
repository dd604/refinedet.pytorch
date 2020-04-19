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
from torch.autograd import Variable
import torch.nn.functional as functional
from libs.utils.net_utils import TCB, make_special_tcb_layer, weights_init
from libs.utils.box_utils import refine_anchors
from libs.modules.anchor_box import AnchorBox
from libs.modules.detect_layer import Detect
from libs.modules.arm_loss import ARMLoss
from libs.modules.odm_loss import ODMLoss
import pdb


class RefineDet(nn.Module):
    """
    """
    def __init__(self, num_classes, cfg):
        super(RefineDet, self).__init__()
        # pdb.set_trace()
        self.num_classes = num_classes
        self.cfg = cfg
        self.anchor_layer = AnchorBox(self.cfg)
        self.is_batchnorm = False
        # anchors are on cpu, their type will be converted accordingly in later
        self.anchors = Variable(self.anchor_layer.forward(), requires_grad=False)
        # vgg backbone
        self.base = None
        self.extra = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layers_out_channels = []
        
        # detection layer and loss layers
        self.detect_layer = Detect(
            self.num_classes, cfg['variance'],
            cfg['top_k_per_class'], cfg['top_k'],
            cfg['detection_conf_threshold'], cfg['detection_nms']
        )
        self.arm_loss_layer = ARMLoss(cfg['gt_overlap_threshold'],
                                      cfg['neg_pos_ratio'],
                                      cfg['variance'])
        self.odm_loss_layer = ODMLoss(
            self.num_classes,
            cfg['gt_overlap_threshold'], cfg['neg_pos_ratio'],
            cfg['variance']
        )
        self.arm_predictions = None
        self.odm_predictions = None
        self.refined_anchors = None
        self.variance = cfg['variance']
        self.pos_anchor_threshold = cfg['pos_anchor_threshold']
        self.ignore_flags_refined_anchor = None

    def _init_modules(self, base_model_path=None, pretrained=True,
                      fine_tuning=True):
        """
        One should overwrite this function and call _init_part_modules()
        """
        raise NotImplementedError('You should overwrite the `_init_modules` '
                                  'function when inheriting this class.')
    
    def _init_part_modules(self):
        # this must be set with cfg
        self.internal_channels = self.cfg['tcb_channles']
        self.pyramid_layer1 = TCB(self.layers_out_channels[0], self.internal_channels,
                             self.internal_channels, self.is_batchnorm)
        self.pyramid_layer2 = TCB(self.layers_out_channels[1], self.internal_channels,
                             self.internal_channels, self.is_batchnorm)
        self.pyramid_layer3 = TCB(self.layers_out_channels[2], self.internal_channels,
                             self.internal_channels, self.is_batchnorm)
        # The pyramid_layer4 has a different constructure
        self.top_in_channels = self.layers_out_channels[3]
        # pdb.set_trace()
        self.pyramid_layer4 = nn.Sequential(*make_special_tcb_layer(
            self.top_in_channels, self.internal_channels, self.is_batchnorm))
        
        # arm and odm
        self._create_arm_head()
        self._create_odm_head()
        
    def _init_weights(self):
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        #pdb.set_trace()
        self.extra.apply(weights_init)
        self.pyramid_layer1.apply(weights_init)
        self.pyramid_layer2.apply(weights_init)
        self.pyramid_layer3.apply(weights_init)
        self.pyramid_layer4.apply(weights_init)
    
        self.arm_loc.apply(weights_init)
        self.arm_conf.apply(weights_init)
        self.odm_loc.apply(weights_init)
        self.odm_conf.apply(weights_init)
        
    def create_architecture(self, base_model_path=None, pretrained=True,
                            fine_tuning=True):
        """
        Firstly, create modules and load pretrained weights for base model if necessary.
        Then initialize weights for heads of detection.
        :param base_model_path: model path for a base network.
        :param pretrained: whether load a pretrained model or not.
        :param fine_tuning: whether fix parameters for a base model.
        :return:
        """
        #pdb.set_trace()
        self._init_modules(base_model_path, pretrained, fine_tuning)
        self._init_weights()
    
    def load_weights(self, model_path):
        """
        Load trained model
        :param model_path: model path.
        :return:
        """
        self.load_state_dict(torch.load(model_path))
        
    def forward(self, x, targets=None):
        """
        Apply network on input.
        :param x: input variable (batch_size, height, width, channel)
        :param targets: targets must be assigned a none variable if training,
        other wise for testing.
        :return: four losses (refer to function self.calculate_loss) for training.
        detection results (refer to function handle self.detect_layer) for testing.
        """

        # forward features
        forward_features = self._get_forward_features(x)
        # print([k.shape for k in forward_features])
        (c1, c2, c3, c4) = forward_features
        # pyramid features
        # Do not overwrite c4
        p4 = self.pyramid_layer4(c4)
        p3 = self.pyramid_layer3(c3, p4)
        p2 = self.pyramid_layer2(c2, p3)
        p1 = self.pyramid_layer1(c1, p2)
        pyramid_features = [p1, p2, p3, p4]
        # print([k.shape for k in pyramid_features])
        if x.is_cuda:
            self.anchors = self.anchors.cuda()
        # Predictions of two heads
        self.arm_predictions = tuple(self._forward_arm_head(forward_features))
        self.odm_predictions = tuple(self._forward_odm_head(pyramid_features))
        # Refine anchores and get ignore flags for refined anchores using self.arm_predictions
        self._refine_arm_anchors(self.arm_predictions)
        if not self.training:
            return self.detect_layer(self.odm_predictions,
                                     self.refined_anchors,
                                     self.ignore_flags_refined_anchor)
        elif targets is not None:
            return self.calculate_loss(targets)
        
    
    def calculate_loss(self, targets):
        """
        Calculate four losses for ARM and ODM when training.
        :param targets: ground truth, shape is (batch_size, num_gt, 5)
        :return arm_loss_loc: location loss for ARM
        :return arm_loss_conf: binary confidence loss for ARM
        :return odm_loss_loc: location loss for ODM
        :return odm_loss_conf: multiple confidence loss for ARM
        """
        arm_loss_loc, arm_loss_conf = self.arm_loss_layer(
            self.arm_predictions, self.anchors, targets)
        odm_loss_loc, odm_loss_conf = self.odm_loss_layer(
            self.odm_predictions, self.refined_anchors,
            self.ignore_flags_refined_anchor, targets)
        # odm_loss_loc = torch.Tensor([1])
        # odm_loss_conf = torch.Tensor([1])
        return arm_loss_loc, arm_loss_conf, odm_loss_loc, odm_loss_conf
    
    def _get_forward_features(self, x):
        """
        One should re-write this function and calculate forward features of layer1-4
        """
        raise NotImplementedError('You should re-write the '
                                  '"_get_forward_features" function')
    
    def _refine_arm_anchors(self, arm_predictions):
        """
        Refine anchores and get ignore flag for refined anchores using outputs of ARM.
        """
        arm_loc, arm_conf = arm_predictions
        # Detach softmax of confidece predictions to block backpropation.
        arm_score = functional.softmax(arm_conf.detach(), -1)
        # Adjust anchors with arm_loc.
        # The refined_pirors is better to be considered as predicted ROIs,
        # like Faster RCNN in a sence.
        self.refined_anchors = refine_anchors(arm_loc.data.detach(), self.anchors.data.detach(),
                                              self.variance)
        self.ignore_flags_refined_anchor = arm_score[:, :, 1] < self.pos_anchor_threshold
        
    def _forward_arm_head(self, forward_features):
        """
        Apply ARM heads to forward features and concatenate results of all heads
        into one variable.
        :param forward_features: a list, features of four layers.
        :return arm_loc_pred: location predictions for layers,
            shape is (batch, num_anchors, 4)
        :return arm_conf_pred: confidence predictions for layers,
            shape is (batch, num_anchors, 2)
        """
        arm_loc_pred = []
        arm_conf_pred = []
        num_classes = 2
        # Apply ARM heads to forward_features and concatenate results
        for (x, l, c) in zip(forward_features, self.arm_loc, self.arm_conf):
            arm_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        # (batch, num_anchors*pred)
        arm_loc_pred = torch.cat([o.view(o.size(0), -1)
                                 for o in arm_loc_pred], 1)
        arm_conf_pred = torch.cat([o.view(o.size(0), -1)
                                  for o in arm_conf_pred], 1)
        arm_loc_pred = arm_loc_pred.view(arm_loc_pred.size(0), -1, 4)
        arm_conf_pred = arm_conf_pred.view(arm_conf_pred.size(0), -1, num_classes)
        
        return arm_loc_pred, arm_conf_pred

    def _forward_odm_head(self, pyramid_features):
        """
        Apply ODM heads to pyramid features and concatenate results of all heads
        into one variable.
        :param pyramid_features: a list, features of four layers.
        :return odm_loc_pred: location predictions for layers,
            shape is (batch, num_anchors, 4)
        :return odm_conf_pred: confidence predictions for layers,
            shape is (batch, num_anchors, num_classes)
        """
        odm_loc_pred = []
        odm_conf_pred = []
        # Apply ODM heads to pyramid features and concatenate results.
        for (x, l, c) in zip(pyramid_features, self.odm_loc, self.odm_conf):
            odm_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc_pred = torch.cat([o.view(o.size(0), -1)
                                    for o in odm_loc_pred], 1)
        odm_conf_pred = torch.cat([o.view(o.size(0), -1)
                                     for o in odm_conf_pred], 1)
        # Shape is (batch, num_anchors, 4)
        odm_loc_pred = odm_loc_pred.view(odm_loc_pred.size(0), -1, 4)
        # Shape is (batch, num_anchors, num_classes)
        odm_conf_pred = odm_conf_pred.view(odm_conf_pred.size(0), -1,
                                               self.num_classes)
        
        return odm_loc_pred, odm_conf_pred
    
    def _create_arm_head(self):
        """
        ARM(Object Detection Module)
        """
        loc_layers = []
        conf_layers = []
        num_classes = 2
        # Relu module in self.layer# does not have 'out_channels' attribution,
        # so we must supply layers_out_channles as inputs for 'Conv2d'
        assert len(self.layers_out_channels) == len(self.cfg['mbox']), \
                'Length of layers_out_channels must match length of cfg["mbox"]'
        for k, v in enumerate(self.layers_out_channels):
            loc_layers += [nn.Conv2d(v, self.cfg['mbox'][k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v, self.cfg['mbox'][k] * num_classes,
                                      kernel_size=3, padding=1)]
        
        self.arm_loc = nn.ModuleList(loc_layers)
        self.arm_conf = nn.ModuleList(conf_layers)
        
    def _create_odm_head(self):
        """
        ODM(Object Detection Module)
        """
        loc_layers = []
        conf_layers = []
        num_classes = self.num_classes
        # internal_channels
        for k in range(len(self.layers_out_channels)):
            loc_layers += [nn.Conv2d(self.internal_channels,
                                     self.cfg['mbox'][k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.internal_channels,
                                      self.cfg['mbox'][k] * num_classes,
                                      kernel_size=3, padding=1)]
        
        self.odm_loc = nn.ModuleList(loc_layers)
        self.odm_conf = nn.ModuleList(conf_layers)
        
