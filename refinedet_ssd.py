import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class TCB(nn.Module):
  """
  Transfer Connection Block Architecture
  This block
  """
  def __init__(self, in_channels):
    intern_channels = 256
    self.conv1 = nn.Conv2d(in_channels, intern_channels, kernel_size=3,
                           padding=1)
    self.bn1 = nn.BatchNorm2d(intern_channels)
    #
    self.conv2 = nn.Conv2d(intern_channels, intern_channels, kernel_size=3,
                           padding=1)
    self.deconv1 = nn.ConvTranspose2d(in_channels, intern_channels,
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
    # batch normalization after element-wise addition
    self.bn2 = nn.BatchNorm2d(intern_channels)
    # self.deconv = F.conv_transpose2d()
    self.conv3 = nn.Conv2d(intern_channels, intern_channels, kernel_size=3,
                           padding=1)
    self.bn3 = nn.BatchNorm2d(intern_channels)
    
  def forward(self, x, next, is_deconv=True):
    conv_branch = nn.ReLU(self.bn1(self.conv1(x)))
    
    # element-wise addation
    if is_deconv:
      combine = self.conv2(conv_branch) + self.deconv1(next)
      combine = nn.ReLU(self.bn2(self.conv2(combine)))
    else:
      combine = conv_branch
    
    combine = nn.ReLU(self.bn3(self.conv3(combine)))
    
    return combine
    

class RefineDet_SSD(nn.Module):
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
      base: VGG16 layers for input, size of either 300 or 500
      extras: extra layers that feed to multibox loc and conf layers
      head: "multibox head" consists of loc and conf conv layers
  """

  def __init__(self, phase, size, base, extras, biclass_head,
               back_pyramid_layers, multiclass_head,
               num_classes):
    super(RefineDet_SSD, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.cfg = (coco, voc)[num_classes == 21]
    self.priorbox = PriorBox(self.cfg)
    self.priors = Variable(self.priorbox.forward(), volatile=True)
    self.size = size

    # SSD network
    self.vgg = nn.ModuleList(base)
    # Layer learns to scale the l2 normalized features from conv4_3
    self.L2Norm = L2Norm(512, 20)
    # module for ARM and multiclassfi
    self.extras = nn.ModuleList(extras)
    self.back_pyramid = nn.ModuleList(back_pyramid_layers)
    
    # for ARM, binary classification.
    self.bi_loc = nn.ModuleList(biclass_head[0])
    self.bi_conf = nn.ModuleList(biclass_head[1])
    
    # for multiple classes classification and regression
    self.multi_loc = nn.ModuleList(multiclass_head[0])
    self.multi_conf = nn.ModuleList(multiclass_head[1])

    if phase == 'test':
      self.bi_softmax = nn.Softmax(dim=-1)
      self.bi_detect = Detect(2, 0, 200, 0.01, 0.45)
      
      self.multi_softmax = nn.Softmax(dim=-1)
      self.multi_detect = Detect(num_classes, 0, 200, 0.01, 0.45)

  def forward(self, x):
    """Applies network layers and ops on input image(s) x.

    Args:
      x: input image or batch of images. Shape: [batch,3,300,300].

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
    
    
    arm_source, bi_output = _arm(x)
    multi_output = _odm(arm_source)
    
    return bi_output, multi_output
  
  def _arm(self, x):
    # arm, anchor refinement moduel
    arm_sources = list()
    bi_loc_pred = list()
    bi_conf_pred = list()
    # apply vgg up to conv4_3 relu
    for k in range(23):
      x = self.vgg[k](x)

    s = self.L2Norm(x)
    arm_sources.append(s)

    # apply vgg up to fc7
    for k in range(23, len(self.vgg)):
      x = self.vgg[k](x)
    arm_sources.append(x)

    # apply extra layers and cache source layer outputs
    for k, v in enumerate(self.extras):
      x = F.relu(v(x), inplace=True)
      if k % 2 == 1:
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
        self.priors
      )
    
    return arm_sources, bi_output
  
  def _odm(self, arm_sources):
    # odm, object detection model
    odm_sources = list()
    multi_loc_pred = list()
    multi_conf_pred = list()
  
    # for odm
    for k in range(len(arm_sources), 0, -1):
      if k == (len(arm_sources) - 1):
        odm_sources.append(self.back_pyramid[k](arm_sources[k],
                                                arm_sources[k],
                                                is_deconv=False))
      else:
        odm_sources.append(self.back_pyramid[k](arm_sources[k],
                                                arm_sources[k + 1],
                                                is_deconv=True))
    odm_sources.reverse()
    for (x, l, c) in zip(odm_sources, self.multi_loc, self.multi_conf):
      multi_loc_pred.append(l(x).permute(0, 2, 3, 1).contiguous())
      multi_conf_pred.append(c(x).permute(0, 2, 3, 1).contiguous())
    multi_loc_pred = torch.cat([o.view(o.size(0), -1)
                                for o in multi_loc_pred], 1)
    multi_conf_pred = torch.cat([o.view(o.size(0), -1)
                                 for o in multi_conf_pred], 1)
  
    if self.phase == "test":
      multi_output = self.detect(
        multi_loc_pred.view(multi_loc_pred.size(0), -1, 4),
        self.softmax(multi_conf_pred.view(multi_conf_pred.size(0), -1, 2)),
        self.priors.type(type(x.data))
      )
    else:
      multi_output = (
        multi_loc_pred.view(multi_loc_pred.size(0), -1, 4),
        multi_conf_pred.view(multi_conf_pred.size(0), -1, 2),
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
    for v in cfg:
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
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def odm_multibox(back_pyramid_layers, cfg, num_classes):
  """
  Multiclass
  :param back_pyramid_layers:
  :param cfg:
  :param num_classes:
  :return:
  """
  loc_layers = []
  conf_layers = []
  # vgg_source = [21, 2]
  
  for k, v in enumerate(back_pyramid_layers):
    loc_layers += [nn.Conv2d(v.out_channels,
                             cfg[k] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(v.out_channels,
                              cfg[k] * num_classes, kernel_size=3,
                              padding=1)]
    return loc_layers, conf_layers

def arm_bibox(vgg, extra_layers, cfg, num_classes=2):
  """
  binary classification.
  :param vgg:
  :param extra_layers:
  :param cfg:
  :param num_classes:
  :return:
      vgg, extra_layers, (loc_layers, conf_layers)
  """
  loc_layers = []
  conf_layers = []
  vgg_source = [21, -2]
  for k, v in enumerate(vgg_source):
    loc_layers += [nn.Conv2d(vgg[v].out_channels,
                             cfg[k] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(vgg[v].out_channels,
                      cfg[k] * num_classes, kernel_size=3, padding=1)]
  for k, v in enumerate(extra_layers[1::2], 2):
    loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                              * num_classes, kernel_size=3, padding=1)]
  return vgg, extra_layers, (loc_layers, conf_layers)
  

# from down to up
def add_back_pyramid(cfg):
  layers = []
  for in_channel in cfg:
    layers += [TCB(in_channel)]
  
  return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

# conv4_3
# fc7(conv7_2)
# conv8_2
# conv9_2
# conv10_2
# conv11_2
back_pyramid = {
    '300': [512, 256, 128, 128, 128, 256],
    '512': [],
}

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
