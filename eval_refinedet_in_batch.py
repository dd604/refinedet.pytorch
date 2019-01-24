from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.networks.resnet_refinedet import ResNetRefineDet
from libs.dataset.config import voc320, voc512, coco320, coco512, MEANS
from libs.dataset.transform import detection_collate
from libs.dataset.roidb import combined_roidb
from libs.dataset.blob_dataset import BlobDataset

import pdb

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(
    description='RefineDet Test With Pytorch')
parser.add_argument('--dataset', default='pascal_voc_0712',
                    choices=['pascal_voc', 'pascal_voc_0712', 'coco'],
                    type=str, help='pascal_voc, pascal_voc_0712 or coco')
parser.add_argument('--network', default='vgg16',
                    help='Pretrained base model')
parser.add_argument('--input_size', default=320, type=int,
                    help='Input size for training')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--model_path', default=None, type=str,
                    help='Checkpoint state_dict file to test from')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
args = parser.parse_args()

args.input_size = 320
args.dataset = 'coco'
args.network = 'vgg16'
postfix_iter = 400000
args.model_path = './weights/{}/refinedet{}_{}_{}.pth'.format(
    args.network, str(args.input_size), args.dataset,
    str(postfix_iter)
)

num_gpus = 1
if torch.cuda.is_available():
    print('CUDA devices: ', torch.cuda.device)
    print('GPU numbers: ', torch.cuda.device_count())
    num_gpus = torch.cuda.device_count()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print('WARNING: It looks like you have a CUDA device, but are not' +
              'using CUDA.\nRun with --cuda for optimal training speed.')
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def eval_net():
    # Assign imdb_name and imdbval_name according to args.dataset.
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
    # Import config
    if args.dataset == 'coco':
        cfg = (coco320, coco512)[args.input_size == 512]
    elif args.dataset in ['pascal_voc', 'pascal_voc_0712']:
        cfg = (voc320, voc512)[args.input_size == 512]
    # Create imdb, roidb and blob_dataset
    print('Create or load an evaluted imdb.')
    imdb, roidb = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))
    
    blob_dataset = BlobDataset(
        imdb, roidb, transform=SSDAugmentation(cfg['min_dim'], MEANS),
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
        if num_gpus > 1:
            net = torch.nn.DataParallel(refinedet)
        else:
            net = refinedet.cuda()
        cudnn.benchmark = True
    # Load weights
    net.load_weights(args.model_path)
    
    net.eval()
    print('Test RefineDet on:', args.imdbval_name)
    print('Using the specified args:')
    print(args)
    
    data_loader = data.DataLoader(blob_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=0,
                                  shuffle=False,
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    num_images = len(imdb.image_index)
    # num_images = len(blob_dataset)
    num_classes = imdb.num_classes
    # num_object_classes + 1 ?
    print(num_classes)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    
    output_dir = get_output_dir(imdb, save_name)
    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    
    # pdb.set_trace()
    for ibatch, (images, targets) in enumerate(data_loader):
        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)
        
        # timers forward
        _t['im_detect'].tic()
        detections = net(images).data
        detect_time = _t['im_detect'].toc(average=True)
        print('timer: %.4f sec.' % (t1 - t0))
        
        for idx_cur_batch in xrange(args.batch_size):
            idx = ibatch * args.batch_size + idx_cur_batch
            # scale each detection back up to the image
            img = images[idx_cur_batch].data.cpu().numpy()
            h, w = img.shape[0], img.shape[1]
            
            # skip jc = 0, because it's the background class
            for jc in range(1, num_classes):
                dets = detections[idx_cur_batch, jc, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() > 0:
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
            
            print('im_detect: {:d}/{:d} {:.3f}s\n'.format(
                i + 1, num_images, detect_time))
    
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    eval_net()
