# encode: utf-8


import os
import sys
import pdb
# pdb.set_trace()
root_dir = os.path.dirname(__file__)
# Add lib to PYTHONPATH
lib_path = os.path.join(root_dir, '../../')
sys.path.append(lib_path)


import time
import argparse
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from libs.utils.augmentations import SSDAugmentation
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.dataset.config import voc, coco, MEANS
from libs.dataset.coco import COCO_ROOT, COCODetection, COCO_CLASSES
from libs.dataset.voc0712 import VOC_ROOT, VOCDetection, \
    VOCAnnotationTransform
from libs.dataset import *
from torch.autograd import Variable
from libs.utils.net_utils import TCB, weights_init
from libs.modules.prior_box import PriorBox
from libs.modules.detect_layer import Detect
from libs.modules.arm_loss import ARMLoss
from libs.modules.odm_loss import ODMLoss

from libs.utils.box_utils import refine_priors, match, match_and_encode, \
    encode, decode, point_form

import pdb

cfg = voc

prior_layer = PriorBox(cfg)
priors = prior_layer.forward()
# pdb.set_trace()
img_size = prior_layer.image_size
# priors = priors * img_size
pdb.set_trace()
one_priors = torch.unsqueeze(priors.clone(), 0)
batch_size = 8
batch_data = one_priors.repeat(batch_size, 1, 1)
loc_preds = torch.zeros_like(batch_data)
for ind in xrange(batch_size):
    boxes = point_form(batch_data[ind])
    loc_preds[ind] = encode(boxes, priors, cfg['variance'])
    
# for ind in xrange(batch_size):
#     cur_preds = loc_preds[ind]
refined_priors = refine_priors(loc_preds, priors, cfg['variance'])

flag = torch.lt(torch.abs(torch.add(refined_priors, -batch_data)), 1e-6)
flag_sum = flag.all()

# flag = refined_priors != batch_data

print(refined_priors[0, -10:, :])
print(batch_data[3, -10:, :])
print(flag_sum)

