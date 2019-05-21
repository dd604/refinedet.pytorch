import os
import time
import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from libs.utils.augmentations import SSDAugmentation
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.networks.resnet_refinedet import ResNetRefineDet
from libs.dataset.config import voc320, voc512, coco320, coco512, MEANS
from libs.dataset.transform import detection_collate
from libs.dataset.roidb import combined_roidb
from libs.dataset.blob_dataset import BlobDataset
import numpy as np
import random
import pdb

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
setup_seed(0)

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(
    description='RefineDet Training With Pytorch')
parser.add_argument('--dataset', default='voc',
                    choices=['voc', 'coco'],
                    type=str, help='voc or coco')
parser.add_argument('--network', default='vgg16',
                    help='Pretrained base model')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
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
parser.add_argument('--save_folder', default='weights/vgg16',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    print('CUDA devices: ', torch.cuda.device)
    print('GPU numbers: ', torch.cuda.device_count())
    num_gpus = torch.cuda.device_count()
    
num_gpus = 1

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print('WARNING: It looks like you have a CUDA device, but are not' +
              'using CUDA.\nRun with --cuda for optimal training speed.')
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

    
def train():
    # Assign imdb_name and imdbval_name according to args.dataset.
    if args.dataset == "voc":
        #args.imdb_name = "voc_2007_trainval"
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
    refinedet.create_architecture(
        os.path.join(args.save_folder, args.basenet), pretrained=True,
        fine_tuning=True)
    #pdb.set_trace()
    # For CPU
    net = refinedet
    # For GPU/GPUs
    if args.cuda:
        if num_gpus > 1:
            net = torch.nn.DataParallel(refinedet)
        else:
            net = refinedet.cuda()
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

    step_index = 0
    str_input_size = str(cfg['min_dim'])
    model_info = 'refinedet{0}_'.format(str_input_size) + args.dataset
    model_save_folder = os.path.join(args.save_folder, model_info)
    if not os.path.exists(model_save_folder):
        os.mkdir(model_save_folder)
        
    data_loader = data.DataLoader(blob_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # Create batch iterator
    # Number of iterations in each epoch
    num_iter_per_epoch = len(blob_dataset) // args.batch_size
    # number of epoch, in case of resuming from args.start_iter
    num_epoch = (cfg['max_iter'] - args.start_iter) // num_iter_per_epoch
    iteration = args.start_iter
    arm_loss_loc = 0
    arm_loss_conf = 0
    odm_loss_loc = 0
    odm_loss_conf = 0
    for epoch in range(0, num_epoch):
        # pdb.set_trace()
        for i_batch, (images, targets) in enumerate(data_loader):
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
    
            if args.cuda:
                images = Variable(images.cuda())
                targets = Variable(targets.cuda())
            else:
                images = Variable(images)
                targets = Variable(targets)
            # forward
            t0 = time.time()
            # backprop
            optimizer.zero_grad()
            bi_loss_loc, bi_loss_conf, multi_loss_loc, multi_loss_conf = \
                net(images, targets)
            loss = bi_loss_loc.mean() + bi_loss_conf.mean() + \
                   multi_loss_loc.mean() + multi_loss_conf.mean()
            loss.backward()
            optimizer.step()
            t1 = time.time()
            if num_gpus > 1:
                arm_loss_loc = bi_loss_loc.mean().data[0]
                arm_loss_conf = bi_loss_conf.mean().data[0]
                odm_loss_loc = multi_loss_loc.mean().data[0]
                odm_loss_conf = multi_loss_conf.mean().data[0]
            else:
                arm_loss_loc = bi_loss_loc.data[0]
                arm_loss_conf = bi_loss_conf.data[0]
                odm_loss_loc = multi_loss_loc.data[0]
                odm_loss_conf = multi_loss_conf.data[0]
            
            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) +
                      (' || ARM Loss Loc: %.4f  || ARM Loss Conf: %.4f' + 
                       ' || ODM Loss Loc: %.4f  || ODM Loss Conf: %.4f' +
                       ' || Loss: %.4f ||') % (
                          arm_loss_loc, arm_loss_conf, 
                          odm_loss_loc, odm_loss_conf, 
                          loss.data[0]) + ' ')
#                 print('iter ' + repr(iteration) +
#                       ' || Loss: %.4f ||' % (loss.data[0]) + ' ')
                # print('iter ' + repr(iteration) +
                #       ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
    
            if iteration != 0 and iteration % 10000 == 0:
            #if iteration != 0 and iteration % cfg['checkpoint_step'] == 0:
                print('Saving state, iter:', iteration)
                torch.save(refinedet.state_dict(),
                           os.path.join(model_save_folder,
                                        model_info + '_' +
                                        repr(iteration) + '.pth'))

            iteration += 1
        
    torch.save(refinedet.state_dict(),
               os.path.join(args.save_folder, args.dataset + '.pth'))
            
            


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
