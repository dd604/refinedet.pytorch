from __future__ import print_function
import sys
import os
import argparse
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.networks.resnet_refinedet import ResNetRefineDet
from libs.utils.config import voc320, voc512, coco320, coco512, MEANS
from libs.data_layers.transform import detection_collate, BaseTransform
from libs.data_layers.roidb import combined_roidb, get_output_dir
from libs.data_layers.blob_dataset import BlobDataset


import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
        
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(
    description='RefineDet Test With Pytorch')
parser.add_argument('--dataset', default='voc', choices=['voc', 'coco'],
                    type=str, help='voc or coco')
parser.add_argument('--network', default='vgg16',
                    help='Base network')
parser.add_argument('--input_size', default=320, type=int,
                    help='Input size for evaluation')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for evaluation')
parser.add_argument('--model_path', default=None, type=str,
                    help='Checkpoint state_dict file to test from')
parser.add_argument('--result_path', default='./detection_output', type=str,
                    help='Path to store detection results in evaluation')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to evaluate model')
args = parser.parse_args()



if torch.cuda.is_available():
    print('CUDA devices: ', torch.cuda.device)
    print('GPU numbers: ', torch.cuda.device_count())
    num_gpus = torch.cuda.device_count()

num_gpus = 1

def eval_net():
    # Assign imdb_name and imdbval_name according to args.dataset.
    if args.dataset == "voc":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
    # Import config
    if args.dataset == 'coco':
        cfg = (coco320, coco512)[args.input_size==512]
    elif args.dataset == 'voc': 
        cfg = (voc320, voc512)[args.input_size==512]
    # Create imdb, roidb and blob_dataset
    print('Create or load an evaluted imdb.')
    imdb, roidb = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))
    blob_dataset = BlobDataset(
        imdb, roidb, transform=BaseTransform(cfg['min_dim'], MEANS),
        target_normalization=True)

    # Construct networks.
    print('Construct {}_refinedet network.'.format(args.network))
    if args.network == 'vgg16':
        refinedet = VGGRefineDet(cfg['num_classes'], cfg)
    elif args.network == 'resnet101':
        refinedet = ResNetRefineDet(cfg['num_classes'], cfg)
    refinedet.create_architecture()
    # For CPU
    net = refinedet
    # For GPU/GPUs
    if args.cuda:
        net = refinedet.cuda()
        if num_gpus > 1:
            net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # Load weights
    net.load_weights(args.model_path)
    net.eval()
    print('Test RefineDet on:', args.imdbval_name)
    print('Using the specified args:')
    print(args)

    num_images = len(imdb.image_index)
    num_classes = imdb.num_classes
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    
    output_dir = get_output_dir(imdb, args.result_path)
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    # set no grad for variables
    torch.set_grad_enabled(False)
    for idx in range(num_images):
        img, gt, h, w = blob_dataset.pull_item(idx)
        input = Variable(img.unsqueeze(0))
        if args.cuda:
            input = input.cuda()
        # timers forward
        _t['im_detect'].tic()
        detection = net(input)
        detect_time = _t['im_detect'].toc(average=True)
        print('im_detect: {:d}/{:d} {:.3f}s'.format(
            idx + 1, num_images, detect_time))
        # skip jc = 0, because it's the background class
        for jc in range(1, num_classes):
            dets = detection[0, jc, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if (len(dets) > 0) and (dets.dim() > 0):
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[jc][idx] = cls_dets
            else:
                all_boxes[jc][idx] = empty_array
    
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    eval_net()
