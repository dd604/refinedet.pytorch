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
                conf shape: torch.size(batch_size, num_priors, 2)
                loc shape: torch.size(batch_size, num_priors, 4)
                priors shape: torch.size(num_priors,4)
            priors: Priors
            targets: Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5]
                (last idx is the label, 0 for background, >0 for target).
            
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
        for idx in xrange(num):
            cur_targets = targets[idx].data
            target_flag = cur_targets[:, -1].data > 0
            valid_targets = target_flag.unsqueeze(
                target_flag.dim()).expand_as(cur_targets).view(
                -1, cur_targets.size()[-1])
            truths = valid_targets[:, :-1]
            labels = torch.ones_like(valid_targets[:, -1])
            # targets[idx, :, :-1].data
            # truths = targets[idx][:, :-1].data
            # Binary classes
            # labels = torch.zeros_like(targets[idx][:, -1].data)
            truths = targets[idx][:, :-1].data
            # Binary classes
            labels = torch.zeros_like(targets[idx][:, -1].data)
            # encode results are stored in loc_t and conf_t
            match(self.overlap_thresh, truths, priors.data, self.variance,
                  labels, loc_t, conf_t, idx)
    
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        # valid indice.
        pos = conf_t > 0
        # pdb.set_trace()
        # Localization Loss (Smooth L1)
        # Shape: [batch, num_priors, 4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_pred).detach()
        # Select postives to compute bounding box loss.
        loc_p = loc_pred[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = functional.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # Mimic MAX_NEGATIVE of caffe-ssd
        # Compute max conf across a batch for selecting negatives with large
        # error confidence.
        batch_conf = conf_pred.view(-1, self.num_classes)
        # Sum up losses of all wrong classes.
        # This loss is only used to select max negatives.
        loss_conf_proxy = log_sum_exp(batch_conf) - batch_conf.gather(
            1, conf_t.view(-1, 1))
        loss_conf_proxy = loss_conf_proxy.view(num, -1)
        # Exclude positives
        loss_conf_proxy[pos] = 0
        # Sort and select max negatives
        # Values in loss_c are not less than 0.
        _, loss_idx = loss_conf_proxy.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # pdb.set_trace()
        num_pos = pos.long().sum(1, keepdim=True)
        # num_neg = torch.clamp(self.neg_pos_ratio * num_pos,
        #                       max=pos.size(1) - num_pos)
        num_neg = torch.min(self.neg_pos_ratio * num_pos, pos.size(1) - num_pos)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Total confidence loss includes positives and negatives.
        pos_idx = pos.unsqueeze(2).expand_as(conf_pred)
        neg_idx = neg.unsqueeze(2).expand_as(conf_pred)
        # Use detach() to block backpropagation of idx
        select_conf_pred_idx = (pos_idx + neg_idx).gt(0).detach()
        select_conf_pred = conf_pred[select_conf_pred_idx].view(
            -1, self.num_classes)
        select_target_idx = (pos + neg).gt(0).detach()
        select_target = conf_t[select_target_idx]
        # Final classification loss
        loss_c = functional.cross_entropy(select_conf_pred, select_target,
                                          size_average=False)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + alpha*Lloc(x,l,g)) / N
        # only number of positives
        total_num = num_pos.data.sum()
        loss_l /= total_num
        loss_c /= total_num
    
        return loss_l, loss_c
