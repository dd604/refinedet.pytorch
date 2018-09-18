# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from libs.utils.box_utils import match, log_sum_exp

import pdb

class ARMLoss(nn.Module):
    """
    """
    def __init__(self, overlap_thresh, neg_pos_ratio, variance):
        super(ARMLoss, self).__init__()
        self.overlap_thresh = overlap_thresh
        self.num_classes = 2
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance
    
    def forward(self, predictions, priors, targets):
        """Binary box and classification Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and priors boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            priors (tensor): Priors
            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_pred, conf_pred = predictions
        num = loc_pred.size(0)
        num_priors = priors.size(0)
        
        # result: match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        if loc_pred.is_cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        
        #     pdb.set_trace()
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            # binary classes
            labels = torch.zeros_like(targets[idx][:, -1].data)
            # encode results are stored in loc_t and conf_t
            match(self.overlap_thresh, truths, priors.data, self.variance,
                  labels, loc_t, conf_t, idx)
        
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        
        pos = conf_t > 0
        # pdb.set_trace()
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_pred)
        loc_p = loc_pred[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = functional.smooth_l1_loss(loc_p, loc_t, size_average=False)
        
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_pred.view(-1, self.num_classes)
        # all wrong loss
        loss_c = log_sum_exp(batch_conf) - \
                 batch_conf.gather(1, conf_t.view(-1, 1))
        
        # change view
        loss_c = loss_c.view(num, -1)
        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        
        # Confidence Loss Including Positive and Negative Examples
        neg_idx = neg.unsqueeze(2).expand_as(conf_pred)
        pos_idx = pos.unsqueeze(2).expand_as(conf_pred)
        # gt(0) ?
        select_conf_pred = conf_pred[(pos_idx + neg_idx).gt(0)].view(
            -1, self.num_classes)
        select_target = conf_t[(pos + neg).gt(0)]
        loss_c = functional.cross_entropy(select_conf_pred, select_target,
                                          size_average=False)
        
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + ��Lloc(x,l,g)) / N
        # only positives ?
        total_num = num_pos.data.sum()
        loss_l /= total_num
        loss_c /= total_num
        # loss_c /= (self.neg_pos_ratio * total_num)
        
        return loss_l, loss_c
