import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from libs.dataset import *
import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from libs.networks.vgg_refinedet import VGGRefineDet
import pdb
# pdb.set_trace()
net = VGGRefineDet(voc['num_classes'], voc)    # initialize SSD
# net = build_refinedet('test', coco, 320, 81)    # initialize SSD

# net.load_weights('../weights/refinedet320_VOC_120000.pth')
weights_path = '../weights/refinedet320_VOC_120000.pth'
weights = torch.load(weights_path)
# pdb.set_trace()
for k, v in weights.items():
    print(k)
    # print(k, v.shape)
# create architecture
net.create_architecture()
net.eval()
net = net.cuda()
# for k, v in weights.state_dict():
# pdb.set_trace()
for k, v in net.state_dict().items():
    print (k, v.shape)
    
# load weights
net.load_state_dict(weights)
# # image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
# %matplotlib inline
# from matplotlib import pyplot as plt
from libs.dataset import VOCDetection, VOC_ROOT, VOCAnnotationTransform


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
