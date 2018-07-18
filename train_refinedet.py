from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import BiBoxLoss, MultiBoxLoss
from refinedet import build_refinedet
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import pdb


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='COCO', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/root/dataset/coco',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
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
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    print('CUDA devices: ', torch.cuda.device)
    print('GPU numbers: ', torch.cuda.device_count())
    
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    # train
    refinedet = build_refinedet('train', cfg['min_dim'], cfg['num_classes'])
    net = refinedet
    if args.cuda:
        # refinedet = refinedet.cuda(device_ids)
        # net = torch.nn.DataParallel(refinedet,
        #         device_ids=device_ids).cuda(device_ids[0])
        net = torch.nn.DataParallel(refinedet).cuda()
        cudnn.benchmark = True
      
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        refinedet.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        refinedet.vgg.load_state_dict(vgg_weights)

  

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        refinedet.extras.apply(weights_init)
        refinedet.bi_loc.apply(weights_init)
        refinedet.bi_conf.apply(weights_init)

        refinedet.back_pyramid.apply(weights_init)
        refinedet.multi_loc.apply(weights_init)
        refinedet.multi_conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    bi_criterion = BiBoxLoss(0.5, True, 0, True, 3, 0.5, args.cuda)
    multi_criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True,
                                   3, 0.5, 0.6, args.cuda)

    net.train()
    # loss counters
    print('Loading the dataset...')
    print('Training RefineDet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'RefinDet.PyTorch on ' + dataset.name
        vis_legend = ['Binary Loc Loss', 'Binary Conf Loss',
                      'Binary Total Loss',
                      'Multiclass Loc Loss', 'Multiclass Conf Loss',
                      'Multiclass Total Loss',
                      'Total Loss']
        
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    # batch_iterator = iter(data_loader)
    # for iteration in range(args.start_iter, cfg['max_iter']):
    # number of iterations in each epoch
    iter_datasets = len(dataset) // args.batch_size
    epoch_size = iter_datasets
    # number of epoch
    epoch_num = cfg['max_iter'] // iter_datasets
    iteration = 0
    bi_loc_loss = 0
    bi_conf_loss = 0
    multi_loc_loss = 0
    multi_conf_loss = 0
    for epoch in range(0, epoch_num):
        if args.visdom and epoch != 0:
            update_vis_plot(epoch, bi_loc_loss, bi_conf_loss,
                            multi_loc_loss, multi_conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            bi_loc_loss = 0
            bi_conf_loss = 0
            multi_loc_loss = 0
            multi_conf_loss = 0
        
        # pdb.set_trace()
        for i_batch, (images, targets) in enumerate(data_loader):
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
    
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()
            bi_out, multi_out, priors = net(images)
            # backprop
            optimizer.zero_grad()
            bi_loss_l, bi_loss_c = bi_criterion(bi_out, priors, targets)
            multi_loss_l, multi_loss_c = multi_criterion(bi_out, multi_out, priors,
                                                         targets)
            loss = bi_loss_l + bi_loss_c + multi_loss_l + multi_loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            bi_loc_loss += bi_loss_l.data[0]
            bi_conf_loss += bi_loss_c.data[0]
            multi_loc_loss += multi_loss_l.data[0]
            multi_conf_loss += multi_loss_c.data[0]
            
            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) +
                      ' || Loss: %.4f ||' % (loss.data[0]) + ' ')
                # print('iter ' + repr(iteration) +
                #       ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
    
            if args.visdom:
                update_vis_plot(iteration, bi_loss_l.data[0], bi_loss_c.data[0],
                                multi_loss_l.data[0], multi_loss_c.data[0],
                                iter_plot, epoch_plot, 'append')
    
            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(refinedet.state_dict(), 'weights/refinedet320_COCO_' +
                           repr(iteration) + '.pth')

            iteration += 1
        
    torch.save(refinedet.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')
            
            


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, len(_legend))).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, bi_loc, bi_conf, multi_loc, multi_conf,
                    window1, window2, update_type, epoch_size=1):
    num_loss_type = 6
    viz.line(
        X=torch.ones((1, num_loss_type)).cpu() * iteration,
        Y=torch.Tensor([bi_loc, bi_conf, bi_loc + bi_conf,
                        multi_loc, multi_conf, multi_loc + multi_conf,
                        bi_loc + bi_conf + multi_loc + multi_conf]
                       ).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, num_loss_type)).cpu(),
            Y=torch.Tensor([bi_loc, bi_conf, bi_loc + bi_conf,
                        multi_loc, multi_conf, multi_loc + multi_conf,
                        bi_loc + bi_conf + multi_loc + multi_conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
