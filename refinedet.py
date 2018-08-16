import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from layers import *
from data import voc, coco
# from config import COCO320_CFG, COCO512_CFG
# from data.config import COCO320_CFG, COCO512_CFG
import os

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
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0)
    # else:
    #     print('Warning, an unknowned instance!!')
    #     print(m)


#

class TCB(nn.Module):
    """
    Transfer Connection Block Architecture
    This block
    """
    
    def __init__(self, lateral_channels, channles,
                 internal_channels=256):
        """
        :param lateral_channels: number of forward feature channles
        :param channles: number of pyramid feature channles
        :param internal_channels: number of internal channels
        """
        super(TCB, self).__init__()
        # conv + bn + relu
        self.conv1 = nn.Conv2d(lateral_channels, internal_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # ((conv2 + bn2) element-wise add  (deconv + deconv_bn)) + relu
        # batch normalization before element-wise addition
        self.conv2 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.deconv = nn.ConvTranspose2d(channles, internal_channels,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.deconv_bn = nn.BatchNorm2d(internal_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # conv + bn + relu
        self.conv3 = nn.Conv2d(internal_channels, internal_channels,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(internal_channels)
        self.relu3 = nn.ReLU(inplace=True)
        
        # attribution
        self.out_channels = internal_channels
    
    def forward(self, lateral, x):
        lateral_out = self.relu1(self.bn1(self.conv1(lateral)))
        # element-wise addation
        out = self.relu2(
            self.bn2(self.conv2(lateral_out)) +
            self.deconv_bn(self.deconv(x))
        )
        
        out = self.relu3(self.bn3(self.conv3(out)))
        
        return out


class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
  
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 320 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """
    
    def __init__(self, phase, size, base_layers, key_ids, extras_layers,
                 arm_head, odm_head, multiple_in_channles, num_classes,
                 negative_prior_threshold=0.99):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        # create priors
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True).cuda()
        # pdb.set_trace()
        self.size = size
        # vgg backbone
        self.base = nn.ModuleList(base_layers)
        self.base_key_ids = key_ids
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm_conv4_3 = L2Norm(512, 8)
        self.L2Norm_conv5_3 = L2Norm(512, 10)
        # module for ARM and multiclassfi
        self.extras = nn.ModuleList(extras_layers)
        
        self.top_in_channels = multiple_in_channles[-1]
        # this must be set with cfg
        self.internal_channels = 256
        top_layers = [nn.Conv2d(self.top_in_channels, self.internal_channels,
                                kernel_size=3, padding=1),
                      nn.BatchNorm2d(self.internal_channels),
                      nn.ReLU(inplace=True)]
        top_layers += ([nn.Conv2d(self.internal_channels, self.internal_channels,
                                  kernel_size=3, padding=1),
                        nn.BatchNorm2d(self.internal_channels),
                        nn.ReLU(inplace=True)] * 2)
        self.pyramid_4 = nn.ModuleList(top_layers)
        self.pyramid_1 = TCB(multiple_in_channles[0], self.internal_channels,
                             self.internal_channels)
        self.pyramid_2 = TCB(multiple_in_channles[1], self.internal_channels,
                             self.internal_channels)
        self.pyramid_3 = TCB(multiple_in_channles[2], self.internal_channels,
                             self.internal_channels)
        
        # for ARM, binary classification.
        self.bi_loc = nn.ModuleList(arm_head[0])
        self.bi_conf = nn.ModuleList(arm_head[1])
        
        # for multiple classes classification and regression
        self.multi_loc = nn.ModuleList(odm_head[0])
        self.multi_conf = nn.ModuleList(odm_head[1])
        # pdb.set_trace()
        self.negative_prior_threshold = negative_prior_threshold
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200,
                                 self.negative_prior_threshold, 0.01, 0.45)
        
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        self.base.apply(weights_init)
        self.extras.apply(weights_init)
        self.pyramid_1.apply(weights_init)
        self.pyramid_2.apply(weights_init)
        self.pyramid_3.apply(weights_init)
        self.pyramid_4.apply(weights_init)
        
        self.bi_loc.apply(weights_init)
        self.bi_conf.apply(weights_init)
        self.multi_loc.apply(weights_init)
        self.multi_conf.apply(weights_init)
    
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
    
        Args:
          x: input image or batch of images. Shape: [batch,3,320,320].
    
        Return:
          Depending on phase:
          test:
              Variable(tensor) of output class label predictions,
              confidence score, and corresponding location predictions for
              each object detected. Shape: [batch,topk,7]
    
          train:
            list of concat outputs from:
              1: confidence layers, Shape: [batch*num_priors,num_classes]
              2: localization layers, Shape: [batch,num_priors*4]
              3: priorbox layers, Shape: [2,num_priors*4]
        """
        
        forward_features = self.base_forward(x)
        bi_output, priors = self._arm_head_forward(forward_features)
        multi_output = self.odm_forward(forward_features)
        if self.phase == 'test':
            return self.detect(bi_output[0], bi_output[1], multi_output[0], \
                               self.softmax(multi_output[1].view(multi_output[1].size(0),
                                                                 -1, self.num_classes)), priors)
        else:
            return bi_output, multi_output, priors

    def base_forward(self, x):
        forward_features = list()
        # apply vgg up to conv4_3 after relu
        # pool4, 23; pool5, 30
        pool4 = self.base_key_ids[0] + 1
        pool5 = self.base_key_ids[1] + 1
        # apply vgg upto conv4_3
        for k in range(pool4):
            x = self.base[k](x)
        s = self.L2Norm_conv4_3(x)
        forward_features.append(s)
        # forward_features.append(x)

        # apply vgg upto conv5_3 after relu
        for k in range(pool4, pool5):
            x = self.base[k](x)
        s = self.L2Norm_conv5_3(x)
        forward_features.append(s)
        # forward_features.append(x)

        # apply vgg up to conv_fc7
        for k in range(pool5, self.base_key_ids[2] + 1):
            x = self.base[k](x)
        forward_features.append(x)

        # apply extra layers and cache source layer outputs
        # conv6_2
        for k, v in enumerate(self.extras):
            x = v(x)
        forward_features.append(x)
        
        return forward_features
    # _arm
    def _arm_head_forward(self, forward_features):
        # arm, anchor refinement moduel
        bi_loc_pred = list()
        bi_conf_pred = list()
        
        #     pdb.set_trace()
        # apply multibox head to source layers
        num_classes = 2
        for (x, l, c) in zip(forward_features, self.bi_loc, self.bi_conf):
            bi_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            bi_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
        # (batch, N*pred)
        bi_loc_pred = torch.cat([o.view(o.size(0), -1)
                                 for o in bi_loc_pred], 1)
        bi_conf_pred = torch.cat([o.view(o.size(0), -1)
                                  for o in bi_conf_pred], 1)
        
        bi_output = (bi_loc_pred.view(bi_loc_pred.size(0), -1, 4),
                     bi_conf_pred.view(bi_conf_pred.size(0), -1, num_classes))
        
        return bi_output, self.priors

    # _odm
    def _odm_forward(self, forward_features):
        # odm, object detection model
        multi_loc_pred = list()
        multi_conf_pred = list()
        
        #     pdb.set_trace()
        # print([cur.size() for cur in arm_sources])
        # for odm
        # reverse
        k_range = range(len(forward_features))
        k_range.reverse()
        (f1, f2, f3, f4) = (forward_features[0], forward_features[1], \
                            forward_features[2], forward_features[3])
        
        p6 = self.pyramid_4(f4)
        p5 = self.pyramid_3(f3, p6)
        p4 = self.pyramid_2(f2, p5)
        p3 = self.pyramid_1(f1, p4)
        
        pyramid_features = [p3, p4, p5, p6]

        # print([cur.size() for cur in pyramid_features])
        # pdb.set_trace()
        for (x, l, c) in zip(pyramid_features, self.multi_loc, self.multi_conf):
            # print(l(x).size())
            # print(c(x).size())
            multi_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
            multi_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
            # print(multi_loc_pred[-1].size())
            # print(multi_conf_pred[-1].size())
        multi_loc_pred = torch.cat([o.view(o.size(0), -1)
                                    for o in multi_loc_pred], 1)
        multi_conf_pred = torch.cat([o.view(o.size(0), -1)
                                     for o in multi_conf_pred], 1)
        
        multi_output = (multi_loc_pred.view(multi_loc_pred.size(0), -1, 4),
                        multi_conf_pred.view(multi_conf_pred.size(0), -1, self.num_classes),)
        
        return multi_output
    
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext in ('.pkl', '.pth'):
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, in_channles, batch_norm=False):
    layers = []
    channels = in_channles
    key_ids = []
    multiple_in_channles = []
    # from conv1_1 to conv5_3
    # index from 0
    # conv4_3, 22
    # conv5_3, 29; pool5, 30
    # to 30 ['M']
    for v in cfg[:-2]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            channels = v
    # relu4_3, relu5_3
    key_ids += [22, 29]
    multiple_in_channles += [layers[k-2].out_channels if batch_norm
                            else layers[k-1].out_channels for k in key_ids]
    # after pool5
    # dilation 6.
    dilation = 3
    kernel_size = 3
    padding = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    # cfg[-3] == 'M'
    conv_fc6 = nn.Conv2d(cfg[-4], cfg[-2], kernel_size=kernel_size, padding=padding,
                         dilation=dilation)
    conv_fc7 = nn.Conv2d(cfg[-2], cfg[-1], kernel_size=1)
    # [31 - 34]
    layers += [conv_fc6, nn.ReLU(inplace=True), conv_fc7, nn.ReLU(inplace=True)]
    assert (len(layers) == 35)
    key_ids += [34]
    # pdb.set_trace()
    # print(layers)
    multiple_in_channles += [layers[-2].out_channels]
    
    return layers, key_ids, multiple_in_channles


def add_extras(cfg, in_channels, multiple_in_channels, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    channels = in_channels
    # conv6_1 and conv6_2
    for k, v in enumerate(cfg):
        if (k == 0):
            conv2d = nn.Conv2d(channels, v, kernel_size=1, stride=1, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        else:
            conv2d = nn.Conv2d(channels, v, kernel_size=3, stride=2, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        channels = v
    
    multiple_in_channels += [layers[-3].out_channels if batch_norm
                             else layers[-2].out_channels]
    
    return layers, multiple_in_channels

# arm_bibox
def add_arm_head(multiple_in_channels, priors_cfg):
    """
    binary classification.
    :param multiple_in_channels:
    :param priors_cfg:
    :return:
        loc_layers, conf_layers
    """
    assert (len(multiple_in_channels) == len(priors_cfg),
            'Number of in-layers must match priors_cfg.')
    loc_layers = []
    conf_layers = []
    num_classes = 2
    for k, v in enumerate(multiple_in_channels):
        loc_layers += [nn.Conv2d(v, priors_cfg[k] * 4,
                                 kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v, priors_cfg[k] * num_classes,
                                  kernel_size=3, padding=1)]
        
    return loc_layers, conf_layers

# odm_multibox
def add_odm_head(multiple_in_channels, priors_cfg, num_classes):
    """
    Multiclass
    :param multiple_in_channels:
    :param priors_cfg:
    :param num_classes:
    :return:
        loc_layers, conf_layers
    """
    loc_layers = []
    conf_layers = []
    # pdb.set_trace()
    for k, v in enumerate(multiple_in_channels):
        loc_layers += [nn.Conv2d(v, priors_cfg[k] * 4,
                                 kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v, priors_cfg[k] * num_classes,
                                  kernel_size=3, padding=1)]
        
    return loc_layers, conf_layers


# from down to up
def add_back_pyramid(cfg):
    layers = []
    for k in range(len(cfg)):
        if k == (len(cfg) - 1):
            layers += [TCB(cfg[k], cfg[k])]
        else:
            layers += [TCB(cfg[k], cfg[k + 1])]
    
    return layers


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
base = {
    '320': [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M',
            1024, 1024],
    
    '512': [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M',
            1024, 1024],
}

# after conv_fc7
# conv6_1, conv6_2
extras = {
    '320': [256, 512],
    # '320': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 512],
}

# conv4_3
# conv5_3
# fc7(conv7_2)
# conv6_2

back_pyramid = {
    '320': [512, 512, 1024, 512],
    '512': [512, 512, 1024, 512],
}


def build_refinedet(phase, cfg, size=320, num_classes=21,
                    negative_prior_threshold=0.99):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD320 (size=320) is supported!")
        return
    
    base_layers, key_ids, multiple_in_channels = vgg(base[str(size)], 3)
    extra_layers, multiple_in_channels = add_extras(extras[str(size)],
                base[str(size)][-1], multiple_in_channels)
    # backbone-agnostic modules
    arm_head = add_arm_head(multiple_in_channels, cfg['mbox'])
    odm_head = add_odm_head(multiple_in_channels, cfg['mbox'], num_classes)
    
    # pdb.set_trace()
    return RefineDet(phase,
                     size,
                     base_layers,
                     key_ids,
                     extra_layers,
                     arm_head,
                     odm_head,
                     multiple_in_channels,
                     num_classes,
                     negative_prior_threshold)
