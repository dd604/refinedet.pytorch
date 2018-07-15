import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os

import pdb


class TCB(nn.Module):
  """
  Transfer Connection Block Architecture
  This block
  """
  def __init__(self, in_channels, next_in_channles):
    super(TCB, self).__init__()
    intern_channels = 256
    self.conv1 = nn.Conv2d(in_channels, intern_channels, kernel_size=3,
                           padding=1)
    self.bn1 = nn.BatchNorm2d(intern_channels)
    #
    self.conv2 = nn.Conv2d(intern_channels, intern_channels, kernel_size=3,
                           padding=1)
    self.deconv1 = nn.ConvTranspose2d(next_in_channles, intern_channels,
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
    # batch normalization after element-wise addition
    self.bn2 = nn.BatchNorm2d(intern_channels)
    # self.deconv = F.conv_transpose2d()
    self.conv3 = nn.Conv2d(intern_channels, intern_channels, kernel_size=3,
                           padding=1)
    self.bn3 = nn.BatchNorm2d(intern_channels)
    self.out_channels = intern_channels
    
  def forward(self, x, next, is_deconv=True):
    conv_branch = F.relu(self.bn1(self.conv1(x)))
    # element-wise addation
    if is_deconv:
      combine = self.conv2(conv_branch) + self.deconv1(next)
      combine = F.relu(self.bn2(combine))
      # combine = F.relu(self.bn2(self.conv2(combine)))
    else:
      combine = conv_branch
      
    combine = F.relu(self.bn3(self.conv3(combine)))
    
    return combine
    

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

  def __init__(self, phase, size, base, key_ids, extras, arm_head,
               back_pyramid_layers, odm_head,
               num_classes):
    super(RefineDet, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.cfg = (coco, voc)[num_classes == 21]
    # create priors
    self.priorbox = PriorBox(self.cfg)
    self.priors = Variable(self.priorbox.forward(), volatile=True).cuda()
    # pdb.set_trace()
    self.size = size

    # SSD network
    self.vgg = nn.ModuleList(base)
    self.vgg_key_ids = key_ids
    # Layer learns to scale the l2 normalized features from conv4_3
    self.L2Norm_conv4_3 = L2Norm(512, 8)
    self.L2Norm_conv5_3 = L2Norm(512, 10)
    # module for ARM and multiclassfi
    self.extras = nn.ModuleList(extras)
    self.back_pyramid = nn.ModuleList(back_pyramid_layers)
    
    # for ARM, binary classification.
    self.bi_loc = nn.ModuleList(arm_head[0])
    self.bi_conf = nn.ModuleList(arm_head[1])
    
    # for multiple classes classification and regression
    self.multi_loc = nn.ModuleList(odm_head[0])
    self.multi_conf = nn.ModuleList(odm_head[1])
    # pdb.set_trace()
    
    if phase == 'test':
      self.bi_softmax = nn.Softmax(dim=-1)
      self.bi_detect = Detect(2, 0, 200, 0.01, 0.45)
      
      self.multi_softmax = nn.Softmax(dim=-1)
      self.multi_detect = Detect(num_classes, 0, 200, 0.01, 0.45)

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
    
    
    arm_source, bi_output, priors = self._arm(x)
    multi_output = self._odm(arm_source)
    
    return bi_output, multi_output, priors
  
  def _arm(self, x):
    # arm, anchor refinement moduel
    arm_sources = list()
    bi_loc_pred = list()
    bi_conf_pred = list()
    # apply vgg up to conv4_3 after relu
    # pool4, 23; pool5, 30
    pool4 = self.vgg_key_ids[0] + 1
    pool5 = self.vgg_key_ids[1] + 1
    # apply vgg upto conv4_3
    for k in range(pool4):
      x = self.vgg[k](x)
    s = self.L2Norm_conv4_3(x)
    arm_sources.append(s)
    
    # apply vgg upto conv5_3 after relu
    for k in range(pool4, pool5):
      x = self.vgg[k](x)
    s = self.L2Norm_conv5_3(x)
    arm_sources.append(s)
    
    # apply vgg up to conv_fc7
    for k in range(pool5, self.vgg_key_ids[2] + 1):
      x = self.vgg[k](x)
    arm_sources.append(x)
    

    # apply extra layers and cache source layer outputs
    # conv6_2
    for k, v in enumerate(self.extras):
      x = v(x)
    arm_sources.append(x)

    # apply multibox head to source layers
    for (x, l, c) in zip(arm_sources, self.bi_loc, self.bi_conf):
      bi_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
      bi_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
    # (batch, N*pred)
    bi_loc_pred = torch.cat([o.view(o.size(0), -1)
                             for o in bi_loc_pred], 1)
    bi_conf_pred = torch.cat([o.view(o.size(0), -1)
                              for o in bi_conf_pred], 1)

    if self.phase == "test":
      bi_output = self.detect(
        bi_loc_pred.view(bi_loc_pred.size(0), -1, 4),
        self.softmax(bi_loc_pred.view(bi_conf_pred.size(0), -1, 2)),
        self.priors.type(type(x.data))
      )
    else:
      bi_output = (
        bi_loc_pred.view(bi_loc_pred.size(0), -1, 4),
        bi_conf_pred.view(bi_conf_pred.size(0), -1, 2),
      )
    
    return arm_sources, bi_output, self.priors
  
  def _odm(self, arm_sources):
    # odm, object detection model
    odm_sources = list()
    multi_loc_pred = list()
    multi_conf_pred = list()
    
    # pdb.set_trace()
    print([cur.size() for cur in arm_sources])
    # for odm
    for k in range(len(arm_sources)-1, -1, -1):
      if k == (len(arm_sources) - 1):
        odm_sources.append(self.back_pyramid[k](arm_sources[k],
                                                arm_sources[k],
                                                is_deconv=False))
      else:
        odm_sources.append(self.back_pyramid[k](arm_sources[k],
                                                arm_sources[k + 1],
                                                is_deconv=True))
    odm_sources.reverse()
    print([cur.size() for cur in odm_sources])
    # pdb.set_trace()
    for (x, l, c) in zip(odm_sources, self.multi_loc, self.multi_conf):
      print(l(x).size())
      print(c(x).size())
      multi_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
      multi_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
      print(multi_loc_pred[-1].size())
      print(multi_conf_pred[-1].size())
    multi_loc_pred = torch.cat([o.view(o.size(0), -1)
                                for o in multi_loc_pred], 1)
    multi_conf_pred = torch.cat([o.view(o.size(0), -1)
                                 for o in multi_conf_pred], 1)
  
    if self.phase == "test":
      multi_output = self.detect(
        multi_loc_pred.view(multi_loc_pred.size(0), -1, 4),
        self.softmax(multi_conf_pred.view(multi_conf_pred.size(0), -1,
                                          self.num_classes)),
        self.priors.type(type(x.data))
      )
    else:
      multi_output = (
        multi_loc_pred.view(multi_loc_pred.size(0), -1, 4),
        multi_conf_pred.view(multi_conf_pred.size(0), -1, self.num_classes),
      )
    
    return multi_output
  
  
  def load_weights(self, base_file):
      other, ext = os.path.splitext(base_file)
      if ext == '.pkl' or '.pth':
          print('Loading weights into state dict...')
          self.load_state_dict(torch.load(base_file,
                               map_location=lambda storage, loc: storage))
          print('Finished!')
      else:
          print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
  layers = []
  in_channels = i
  key_ids = []
  # from conv1_1 to conv5_3
  # index from 0
  # conv4_3, 22
  # conv5_3, 29; pool5, 30
  for v in cfg[:-2]:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    elif v == 'C':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  # relu4_3, relu5_3
  key_ids += [22, 29]
  # after pool5
  # dilation 6.
  dilation = 3
  kernel_size = 3
  padding = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
  conv_fc6 = nn.Conv2d(cfg[-4], cfg[-2], kernel_size=kernel_size, padding=padding,
                       dilation=dilation)
  conv_fc7 = nn.Conv2d(cfg[-2], cfg[-1], kernel_size=1)
  # [31 - 34]
  layers += [conv_fc6, nn.ReLU(inplace=True), conv_fc7, nn.ReLU(inplace=True)]
  assert(len(layers) == 35)
  key_ids += [34]
  # pdb.set_trace()
  # print(layers)
  return layers, key_ids


def add_extras(cfg, i, batch_norm=False):
  # Extra layers added to VGG for feature scaling
  layers = []
  in_channels = i
  # conv6_1 and conv6_2
  for k, v in enumerate(cfg):
    if (k % 2 == 0):
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


def arm_bibox(vgg, vgg_key_ids, extra_layers, priors_cfg):
  """
  binary classification.
  :param vgg:
  :param extra_layers:
  :param cfg:
  :return:
      vgg, extra_layers, (loc_layers, conf_layers)
  """
  loc_layers = []
  conf_layers = []
  num_classes = 2
  # relu has no 'out_channels' attribution.
  for k, v in enumerate(vgg_key_ids):
    loc_layers += [nn.Conv2d(vgg[v-1].out_channels,
                             priors_cfg[k] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(vgg[v-1].out_channels,
                              priors_cfg[k] * num_classes, kernel_size=3, padding=1)]
  
  # for k, v in enumerate(extra_layers[1::2], 2):
  loc_layers += [nn.Conv2d(extra_layers[-2].out_channels, priors_cfg[-1]
                           * 4, kernel_size=3, padding=1)]
  conf_layers += [nn.Conv2d(extra_layers[-2].out_channels, priors_cfg[-1]
                            * num_classes, kernel_size=3, padding=1)]
  
  return vgg, extra_layers, (loc_layers, conf_layers)


def odm_multibox(back_pyramid_layers, priors_cfg, num_classes):
  """
  Multiclass
  :param back_pyramid_layers:
  :param cfg:
  :param num_classes:
  :return:
    back_pyramid_layers, (loc_layers, conf_layers)
  """
  loc_layers = []
  conf_layers = []
  # pdb.set_trace()
  for k, v in enumerate(back_pyramid_layers):
    loc_layers += [nn.Conv2d(v.out_channels,
                             priors_cfg[k] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(v.out_channels,
                              priors_cfg[k] * num_classes, kernel_size=3,
                              padding=1)]
  return back_pyramid_layers, (loc_layers, conf_layers)



# from down to up
def add_back_pyramid(cfg):
  layers = []
  for k in range(len(cfg)):
    if k == (len(cfg) - 1):
      layers += [TCB(cfg[k], cfg[k])]
    else:
      layers += [TCB(cfg[k], cfg[k+1])]

  
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


base = {
    '320': [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M',
            1024, 1024],
    '512': [],
}

# after conv_fc7
# conv6_1, conv6_2
extras = {
    '320': [256, 512],
    # '320': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

# conv4_3
# conv5_3
# fc7(conv7_2)
# conv6_2

back_pyramid = {
    '320': [512, 512, 1024, 512],
    '512': [],
}

mbox = {
    '320': [4, 4, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_refinedet(phase, size=320, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD320 (size=320) is supported!")
        return
    vgg_layers, key_ids = vgg(base[str(size)], 3)
    base_, extras_, arm_head_ = arm_bibox(vgg_layers, key_ids,
                                          add_extras(extras[str(size)], 1024),
                                          mbox[str(size)])
    back_pyramid_layers_, odm_head_ = odm_multibox(
      add_back_pyramid(back_pyramid[str(size)]),
      mbox[str(size)],
      num_classes)
    # pdb.set_trace()
    return RefineDet(phase, size, base_, key_ids, extras_, arm_head_,
                     back_pyramid_layers_, odm_head_, num_classes)
