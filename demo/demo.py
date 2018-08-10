import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from data import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from refinedet import build_refinedet
import pdb

net = build_refinedet('test', voc, 320, 21)    # initialize SSD
# net = build_refinedet('test', coco, 320, 81)    # initialize SSD
net.load_weights('../weights/refinedet320_COCO_5000.pth')

# # image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
# %matplotlib inline
# from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
import pdb
# here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
# img_id = 60
img_id = 62
image = testset.pull_image(img_id)
# pdb.set_trace()
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

x = cv2.resize(image, (320, 320)).astype(np.float32)

# import pdb
# pdb.set_trace()
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
# plt.imshow(x / 255.0)
x = torch.from_numpy(x).permute(2, 0, 1)

xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
pdb.set_trace()
y = net(xx)

print(y)