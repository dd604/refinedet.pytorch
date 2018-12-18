# encoding: utf-8
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from libs.utils.net_utils import TCB, make_special_tcb_layer, weights_init
from libs.modules.prior_box import PriorBox
from libs.modules.detect_layer import Detect
from libs.modules.arm_loss import ARMLoss
from libs.modules.odm_loss import ODMLoss
import pdb
# from data.config import cfg as _cfg


class RefineDet(nn.Module):
    """
    """
    def __init__(self, num_classes, cfg):
        super(RefineDet, self).__init__()
        # pdb.set_trace()
        self.num_classes = num_classes
        self.cfg = cfg
        self.prior_layer = PriorBox(self.cfg)
        self.is_batchnorm = False
        # priors are on cpu, their type will be converted accordingly in later
        self.priors = Variable(self.prior_layer.forward(), volatile=True)
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
            self.num_classes, cfg['variance'], cfg['variance'],
            cfg['top_k'], cfg['pos_prior_threshold'],
            cfg['detection_conf_threshold'], cfg['detection_nms']
        )
        self.arm_loss_layer = ARMLoss(cfg['gt_overlap_threshold'],
                                      cfg['neg_pos_ratio'],
                                      cfg['variance'])
        self.odm_loss_layer = ODMLoss(
            self.num_classes,
            cfg['gt_overlap_threshold'], cfg['neg_pos_ratio'],
            cfg['variance'], cfg['variance'],
            cfg['pos_prior_threshold']
        )
        self.arm_predictions = None
        self.odm_predictions = None

    def _init_modules(self, model_path=None, pretrained=True, fine_tuning=True):
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
        self.pyramid_layer4 = nn.ModuleList(make_special_tcb_layer(
            self.top_in_channels, self.internal_channels, self.is_batchnorm))
        
        # arm and odm
        self._build_arm_head()
        self._build_odm_head()
        
    def _init_weights(self):
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        self.extra.apply(weights_init)
        self.pyramid_layer1.apply(weights_init)
        self.pyramid_layer2.apply(weights_init)
        self.pyramid_layer3.apply(weights_init)
        self.pyramid_layer4.apply(weights_init)
    
        self.arm_loc.apply(weights_init)
        self.arm_conf.apply(weights_init)
        self.odm_loc.apply(weights_init)
        self.odm_conf.apply(weights_init)
        
    def create_architecture(self, model_path=None, pretrained=True,
                            fine_tuning=True):
        self._init_modules(model_path, pretrained, fine_tuning)
        self._init_weights()
    
    def forward(self, x, targets=None):
        """Applies network layers and ops on input image(s) x.
        Args:
          x: input image or batch of images. Shape: [batch,3,320,320].
          targets:
        """
        # forward features
        forward_features = self._get_forward_features(x)
        # print([k.shape for k in forward_features])
        (c1, c2, c3, c4) = forward_features
        # pyramid features
        # Do not overwrite c4
        p4 = self.pyramid_layer4[0](c4)
        for k in xrange(1, len(self.pyramid_layer4)):
            p4 = self.pyramid_layer4[k](p4)
        p3 = self.pyramid_layer3(c3, p4)
        p2 = self.pyramid_layer2(c2, p3)
        p1 = self.pyramid_layer1(c1, p2)
        pyramid_features = [p1, p2, p3, p4]
        # print([k.shape for k in pyramid_features])
        if x.is_cuda:
            self.priors = self.priors.cuda()
        # Predictions of two heads
        self.arm_predictions = tuple(self._forward_arm_head(forward_features))
        self.odm_predictions = tuple(self._forward_odm_head(pyramid_features))
        if not self.training:
            return self.detect_layer(self.arm_predictions, self.odm_predictions,
                                     self.priors.data)
        elif targets is not None:
            return self.calculate_loss(targets)
        
    
    def calculate_loss(self, targets):
        arm_loss_loc, arm_loss_conf = self.arm_loss_layer(
            self.arm_predictions, self.priors, targets)
        odm_loss_loc, odm_loss_conf = self.odm_loss_layer(
            self.arm_predictions, self.odm_predictions, self.priors, targets)
        
        return arm_loss_loc, arm_loss_conf, odm_loss_loc, odm_loss_conf
    
    def _get_forward_features(self, x):
        """
        One should rewrite this function and calculate forward features by layer1-4
        """
        raise NotImplementedError('You should rewrite the "calculate_forward_features" function')
    

    def _forward_arm_head(self, forward_features):
        """
        Apply arm_head to forward_features.
        :param forward_features:
        :return:
        arm_loc_pred
        """
        # arm, anchor refinement moduel
        arm_loc_pred = []
        arm_conf_pred = []
        num_classes = 2
        # Apply arm_head to forward_features and concatenate results into
        # a tensor of shape (batch, num_priors*4 or num_priors*num_classes)
        for (x, l, c) in zip(forward_features, self.arm_loc, self.arm_conf):
            arm_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        # (batch, N*pred)
        arm_loc_pred = torch.cat([o.view(o.size(0), -1)
                                 for o in arm_loc_pred], 1)
        arm_conf_pred = torch.cat([o.view(o.size(0), -1)
                                  for o in arm_conf_pred], 1)
        arm_loc_pred = arm_loc_pred.view(arm_loc_pred.size(0), -1, 4)
        arm_conf_pred = arm_conf_pred.view(arm_conf_pred.size(0), -1, num_classes)
        
        return arm_loc_pred, arm_conf_pred

    def _forward_odm_head(self, pyramid_features):

        odm_loc_pred = []
        odm_conf_pred = []
        # pdb.set_trace()
        # Apply odm_head to pyramid_features and concatenate results into
        # a tensor of shape (batch, num_priors*4 or num_priors*num_classes)
        for (x, l, c) in zip(pyramid_features, self.odm_loc, self.odm_conf):
            odm_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc_pred = torch.cat([o.view(o.size(0), -1)
                                    for o in odm_loc_pred], 1)
        odm_conf_pred = torch.cat([o.view(o.size(0), -1)
                                     for o in odm_conf_pred], 1)
        # Shape of loc_pred (batch, num_priors, 4)
        odm_loc_pred = odm_loc_pred.view(odm_loc_pred.size(0), -1, 4)
        # Shape of conf_pred (batch, num_priors, num_classes)
        odm_conf_pred = odm_conf_pred.view(odm_conf_pred.size(0), -1,
                                               self.num_classes)
        
        return odm_loc_pred, odm_conf_pred
    
    def _build_arm_head(self):
        """
        ARM(Object Detection Module)
        """
        loc_layers = []
        conf_layers = []
        num_classes = 2
        # Relu module in self.layer# does not have 'out_channels' attribution,
        # so we must supply layers_out_channles as inputs for 'Conv2d'
        assert (len(self.layers_out_channels) == len(self.cfg['mbox']),
                'Length of layers_out_channels must match length of cfg["mbox"]')
        for k, v in enumerate(self.layers_out_channels):
            loc_layers += [nn.Conv2d(v, self.cfg['mbox'][k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v, self.cfg['mbox'][k] * num_classes,
                                      kernel_size=3, padding=1)]
        
        self.arm_loc = nn.ModuleList(loc_layers)
        self.arm_conf = nn.ModuleList(conf_layers)
        
    def _build_odm_head(self):
        """
        ODM(Object Detection Module)
        """
        loc_layers = []
        conf_layers = []
        num_classes = self.num_classes
        # internal_channels
        for k in xrange(len(self.layers_out_channels)):
            loc_layers += [nn.Conv2d(self.internal_channels,
                                     self.cfg['mbox'][k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.internal_channels,
                                      self.cfg['mbox'][k] * num_classes,
                                      kernel_size=3, padding=1)]
        
        self.odm_loc = nn.ModuleList(loc_layers)
        self.odm_conf = nn.ModuleList(conf_layers)
        