# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
from libs.utils.net_utils import TCB, weights_init
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
            self.num_classes, cfg['top_k'], cfg['pos_prior_threshold'],
            cfg['detection_conf_threshold'], cfg['detection_nms'],
            cfg['variance'], cfg['variance']
        )
        self.arm_loss_layer = ARMLoss(cfg['gt_overlap_threshold'],
                                      cfg['neg_pos_ratio'],
                                      cfg['variance'])
        self.odm_loss_layer = ODMLoss(
            self.num_classes, cfg['pos_prior_threshold'],
            cfg['gt_overlap_threshold'], cfg['neg_pos_ratio'],
            cfg['variance'], cfg['variance']
        )
        self.bi_predictions = None
        self.multi_predictions = None

    def _init_modules(self, model_path=None, pretrained=True):
        """
        One should rewrite this function and call _init_part_modules()
        """
        raise NotImplementedError('You should rewrite the "_init_modules" function when '
                                  'inheriting this class')
    
    def _init_part_modules(self):
        # this must be set with cfg
        self.internal_channels = self.cfg['tcb_channles']
        self.pyramid_layer1 = TCB(self.layers_out_channels[0], self.internal_channels,
                             self.internal_channels)
        self.pyramid_layer2 = TCB(self.layers_out_channels[1], self.internal_channels,
                             self.internal_channels)
        self.pyramid_layer3 = TCB(self.layers_out_channels[2], self.internal_channels,
                             self.internal_channels)
        # pyramid_layer4 has a different constructure
        self.top_in_channels = self.layers_out_channels[3]
        top_layers = [nn.Conv2d(self.top_in_channels, self.internal_channels,
                                kernel_size=3, padding=1),
                      nn.BatchNorm2d(self.internal_channels),
                      nn.ReLU(inplace=True)]
        # repeat twice
        top_layers += ([nn.Conv2d(self.internal_channels, self.internal_channels,
                                  kernel_size=3, padding=1),
                        nn.BatchNorm2d(self.internal_channels),
                        nn.ReLU(inplace=True)] * 2)
        self.pyramid_layer4 = nn.ModuleList(top_layers)
        
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
    
        self.bi_loc.apply(weights_init)
        self.bi_conf.apply(weights_init)
        self.multi_loc.apply(weights_init)
        self.multi_conf.apply(weights_init)
        
    def create_architecture(self, model_path=None, pretrained=True):
        self._init_modules(model_path, pretrained)
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
        (c1, c2, c3, c4) = (forward_features[0], forward_features[1], \
            forward_features[2], forward_features[3])
        # pyramid features
        # p4 = self.pyramid_layer4(c4)
        # do not overwrite c4
        p4 = self.pyramid_layer4[0](c4)
        for k in xrange(1, len(self.pyramid_layer4)):
            p4 = self.pyramid_layer4[k](p4)
        p3 = self.pyramid_layer3(c3, p4)
        p2 = self.pyramid_layer2(c2, p3)
        p1 = self.pyramid_layer1(c1, p2)
        pyramid_features = [p1, p2, p3, p4]
        # print([k.shape for k in pyramid_features])
        if x.is_cuda:
            self.priors.cuda()
        # heads
        bi_loc_pred, bi_conf_pred = self._forward_arm_head(forward_features)
        multi_loc_pred, multi_conf_pred = self._forward_odm_head(pyramid_features)
        self.bi_predictions = (bi_loc_pred, bi_conf_pred)
        self.multi_predictions = (multi_loc_pred, multi_conf_pred)
        
        if self.training == False:
            return self.detect(self.bi_predictions, self.multi_predictions,
                               self.priors)
        elif targets is not None:
            return self.calculate_loss(targets)
    
    def calculate_loss(self, targets):
        bi_loss_loc, bi_loss_conf = self.arm_loss_layer.forward(
            self.bi_predictions, self.priors, targets
        )
        multi_loss_loc, multi_loss_conf = self.odm_loss_layer.forward(
            self.bi_predictions, self.multi_predictions, self.priors, targets
        )
        
        return bi_loss_loc, bi_loss_conf, multi_loss_loc, multi_loss_conf
    
    def _get_forward_features(self, x):
        """
        One should rewrite this function and calculate forward features by layer1-4
        """
        raise NotImplementedError('You should rewrite the "calculate_forward_features" function')
    
    # arm
    def _forward_arm_head(self, forward_features):
        # arm, anchor refinement moduel
        bi_loc_pred = list()
        bi_conf_pred = list()
        
        # apply binary class head to source layers
        num_classes = 2
        for (x, l, c) in zip(forward_features, self.bi_loc, self.bi_conf):
            bi_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            bi_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        # (batch, N*pred)
        bi_loc_pred = torch.cat([o.view(o.size(0), -1)
                                 for o in bi_loc_pred], 1)
        bi_conf_pred = torch.cat([o.view(o.size(0), -1)
                                  for o in bi_conf_pred], 1)
        bi_loc_pred = bi_loc_pred.view(bi_loc_pred.size(0), -1, 4)
        bi_conf_pred = bi_conf_pred.view(bi_conf_pred.size(0), -1, num_classes)
        
        return bi_loc_pred, bi_conf_pred

    # odm
    def _forward_odm_head(self, pyramid_features):
        # odm, object detection model
        multi_loc_pred = list()
        multi_conf_pred = list()
        # pdb.set_trace()
        for (x, l, c) in zip(pyramid_features, self.multi_loc, self.multi_conf):
            multi_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            multi_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        # (batch, N*pred)
        multi_loc_pred = torch.cat([o.view(o.size(0), -1)
                                    for o in multi_loc_pred], 1)
        multi_conf_pred = torch.cat([o.view(o.size(0), -1)
                                     for o in multi_conf_pred], 1)
        multi_loc_pred = multi_loc_pred.view(multi_loc_pred.size(0), -1, 4)
        multi_conf_pred = multi_conf_pred.view(multi_conf_pred.size(0), -1,
                                               self.num_classes)
        
        return multi_loc_pred, multi_conf_pred
    
    def _build_arm_head(self):
        loc_layers = []
        conf_layers = []
        num_classes = 2
        # relu has no 'out_channels' attribution.
        assert (len(self.layers_out_channels) == len(self.cfg['mbox']),
                'Length of layers_out_channels must match length of cfg["mbox"]')
        for k, v in enumerate(self.layers_out_channels):
            loc_layers += [nn.Conv2d(v, self.cfg['mbox'][k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v, self.cfg['mbox'][k] * num_classes,
                                      kernel_size=3, padding=1)]
        self.bi_conf = nn.ModuleList(conf_layers)
        self.bi_loc = nn.ModuleList(loc_layers)
    
    def _build_odm_head(self):
        loc_layers = []
        conf_layers = []
        num_classes = self.num_classes
        # relu has no 'out_channels' attribution.
        for k in xrange(len(self.layers_out_channels)):
            loc_layers += [nn.Conv2d(self.internal_channels,
                                     self.cfg['mbox'][k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.internal_channels,
                                      self.cfg['mbox'][k] * num_classes,
                                      kernel_size=3, padding=1)]
        self.multi_conf = nn.ModuleList(conf_layers)
        self.multi_loc = nn.ModuleList(loc_layers)
        