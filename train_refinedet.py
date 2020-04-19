import os
import time
import argparse
import torch
import _init_paths
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from libs.utils.augmentations import SSDAugmentation
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.networks.resnet_refinedet import ResNetRefineDet
from libs.utils.config import voc320, voc512, coco320, coco512, MEANS
from libs.data_layers.transform import detection_collate
from libs.data_layers.roidb import combined_roidb
from libs.data_layers.blob_dataset import BlobDataset
from libs.utils.path_config import cfg as path_cfg

import numpy as np
import random
import pdb


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(
    description='RefineDet Training With Pytorch')
parser.add_argument('--dataset', default='voc',
                    choices=['voc', 'coco'],
                    type=str, help='voc or coco')
parser.add_argument('--network', default='vgg16',
                    help='backbone network')
parser.add_argument('--base_model', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--input_size', default=320, type=int,
                    help='Input size for training')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume_checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--output_folder', default='output',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained_folder', default='pretrained_model',
                    help='Directory for saving checkpoint models')

args = parser.parse_args()

num_gpus = 0

if torch.cuda.is_available():
    print('CUDA devices: ', torch.cuda.device)
    print('GPU numbers: ', torch.cuda.device_count())
    num_gpus = torch.cuda.device_count()


# num_gpus = 0
num_gpus = 1

# if torch.cuda.is_available():
#     if args.cuda:
#         torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     if not args.cuda:
#         print('WARNING: It looks like you have a CUDA device, but are not' +
#               'using CUDA.\nRun with --cuda for optimal training speed.')
#         torch.set_default_tensor_type('torch.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')


def train():
    # Assign imdb_name and imdbval_name according to args.dataset.
    if args.dataset == "voc":
        # args.imdb_name = "voc_2007_trainval"
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
    # Import config
    if args.dataset == 'coco':
        cfg = (coco320, coco512)[args.input_size == 512]
    elif args.dataset == 'voc':
        cfg = (voc320, voc512)[args.input_size == 512]
    # Create imdb, roidb and blob_dataset
    print('Create or load an imdb.')
    imdb, roidb = combined_roidb(args.imdb_name)
    blob_dataset = BlobDataset(
        imdb, roidb, transform=SSDAugmentation(cfg['min_dim'], MEANS),
        target_normalization=True)
    
    # Construct networks.
    print('Construct {}_refinedet network.'.format(args.network))
    if args.network == 'vgg16':
        refinedet = VGGRefineDet(cfg['num_classes'], cfg)
    elif args.network == 'resnet101':
        refinedet = ResNetRefineDet(cfg['num_classes'], cfg)
    
    pretrained_model = os.path.join(path_cfg.DATA_DIR, args.pretrained_folder, args.base_model)
    refinedet.create_architecture(pretrained_model, pretrained=True, fine_tuning=True)
    # For CPU
    net = refinedet
    # For GPU/GPUs
    if args.cuda:
        net = refinedet.cuda()
        if num_gpus > 1:
            net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # Resume
    if args.resume_checkpoint:
        print('Resuming training, loading {}...'.format(args.resume_checkpoint))
        net.load_weights(args.resume_checkpoint)
    
    # pdb.set_trace()
    # params = net.state_dict()
    # for k, v in params.items():
    #     print(k)
    #     print(v.shape)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    net.train()
    print('Training RefineDet on:', args.imdb_name)
    print('Using the specified args:')
    print(args)
    
    output_folder = os.path.join(path_cfg.OUTPUT_DIR, args.output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    str_input_size = str(cfg['min_dim'])
    model_info = 'refinedet{}_{}'.format(str_input_size, args.dataset)
    model_output_folder = os.path.join(output_folder, '{}'.format(args.network), model_info)
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)
    
    data_loader = data.DataLoader(blob_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # Create batch iterator
    # Number of iterations in each epoch
    base_batch_size = 32
    batch_multiplier = float(args.batch_size) / base_batch_size
    # number of epoch, in case of resuming from args.start_iter
    # fixed number of epoches, no matter what args.batch_size is
    num_epoch = (cfg['max_iter'] - args.start_iter) // (len(blob_dataset) // base_batch_size)
    base_iteration = args.start_iter
    lr_steps = [int(x / batch_multiplier) for x in cfg['lr_steps']]
    actual_iteration = 0
    decay_step = 0
    # print('num_epoch: {}, batch_multiplier: {}, maximum base_iteration {} actual_iteration: {}')
    for epoch in range(0, num_epoch):
        # pdb.set_trace()
        t0 = time.time()
        for i_batch, (images, targets) in enumerate(data_loader):
            if actual_iteration in lr_steps:
                decay_step += 1
                adjust_learning_rate(optimizer, args.gamma, decay_step)
            
            if args.cuda:
                images = Variable(images.cuda())
                targets = Variable(targets.cuda())
            else:
                images = Variable(images)
                targets = Variable(targets)
            t1_data = time.time()
            # forward and backprop
            optimizer.zero_grad()
            bi_loss_loc, bi_loss_conf, multi_loss_loc, multi_loss_conf = \
                net(images, targets)
            loss = bi_loss_loc.mean() + bi_loss_conf.mean() + \
                   multi_loss_loc.mean() + multi_loss_conf.mean()
            # loss = bi_loss_loc.mean() + bi_loss_conf.mean()
            loss.backward()
            optimizer.step()
            t1 = time.time()
            if num_gpus > 1:
                arm_loss_loc = bi_loss_loc.mean().item()
                arm_loss_conf = bi_loss_conf.mean().item()
                odm_loss_loc = multi_loss_loc.mean().item()
                odm_loss_conf = multi_loss_conf.mean().item()
            else:
                arm_loss_loc = bi_loss_loc.item()
                arm_loss_conf = bi_loss_conf.item()
                odm_loss_loc = multi_loss_loc.item()
                odm_loss_conf = multi_loss_conf.item()
            
            if actual_iteration % 10 == 0:
                print('timer: %.4f sec, data loading timer: %.4f sec' % (t1 - t0, t1_data - t0))
                print('iter ' + repr(actual_iteration) +
                      (' || ARM Loss Loc: %.4f  || ARM Loss Conf: %.4f' +
                       ' || ODM Loss Loc: %.4f  || ODM Loss Conf: %.4f' +
                       ' || Loss: %.4f ||') % (
                          arm_loss_loc, arm_loss_conf,
                          odm_loss_loc, odm_loss_conf,
                          loss.item()) + ' ')
            # save checkpoint.
            if actual_iteration != 0 and actual_iteration % (int(10000 / batch_multiplier)) == 0:
                print('Saving state, iter:', base_iteration)
                torch.save(refinedet.state_dict(),
                           os.path.join(model_output_folder,
                                        '_'.join([args.network, model_info, repr(base_iteration) + '.pth'])))
            # update counts.
            actual_iteration += 1
            base_iteration = int(actual_iteration * batch_multiplier)
            t0 = time.time()
            
    torch.save(refinedet.state_dict(),
               os.path.join(model_output_folder, '_'.join([args.network, model_info + '.pth'])))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
