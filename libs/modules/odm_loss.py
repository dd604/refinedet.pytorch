# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from libs.utils.box_utils import log_sum_exp, match, \
    refine_priors

import pdb

class ODMLoss(nn.Module):
    """
    """
    
    def __init__(self, num_classes, overlap_thresh,
                 neg_pos_ratio, arm_variance, variance,
                 pos_prior_threshold):
        super(ODMLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.arm_variance = arm_variance
        self.variance = variance
        self.pos_prior_threshold = pos_prior_threshold
    
    def forward(self, arm_predictions, odm_predictions, priors, targets):
        """Multibox Loss
        :param arm_predictions: tuple (arm_loc_pred, arm_conf_pred),
            arm_loc_pred (batch_size, num_priors, 4),
            arm_conf_pred (batch_size, num_priors, 2)
        :param odm_predictions: tuple (odm_loc_pred, odm_conf_pred)
        :param priors: (num_priors, 4)
        :param targets: Ground truth boxes and labels for a batch,
        shape, [batch_size,num_objs,5]
        (last idx is the label, 0 for background, >0 for target).
        """
        arm_loc = arm_predictions[0]
        # Detach softmax of confidece predictions to block backpropation.
        arm_score = functional.softmax(arm_predictions[1].detach(), -1)
        (loc_pred, conf_pred) = odm_predictions
        num = loc_pred.size(0)
        num_priors = priors.size(0)
        
        # Adjust priors with arm_loc.
        # The refined_pirors is better to be considered as predicted ROIs,
        # like Faster RCNN in a sence.
        refined_priors = refine_priors(arm_loc.data, priors.data,
                                       self.arm_variance)
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        if arm_loc.is_cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        
        # Match refined_priors (predicted ROIs) and ground truth boxes
        # Consider each image in one batch.
        pdb.set_trace()
        for idx in range(num):
            cur_targets = targets[idx].data
            # Ingore background (label id is 0)
            target_flag = cur_targets[:, -1] > 0
            target_flag = target_flag.unsqueeze(
                target_flag.dim()).expand_as(
                cur_targets).contiguous().view(
                -1, cur_targets.size()[-1])
            valid_targets = cur_targets[target_flag].contiguous().view(
                -1, cur_targets.size()[-1])
            truths = valid_targets[:, :-1]
            labels = valid_targets[:, -1]
            
            # truths = targets[idx][:, :-1].data
            # labels = targets[idx][:, -1].data
            # Refined priors of this idx
            cur_priors = refined_priors[idx]
            match(self.overlap_thresh, truths, cur_priors, self.variance,
                  labels, loc_t, conf_t, idx)
        
        # Wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        # Positives
        pos = conf_t > 0
        # Expand as flag
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_pred).detach()
        # Select postives to compute bounding box loss.
        loc_p = loc_pred[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = functional.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_pred.view(-1, self.num_classes)
        # Sum up losses of all wrong classes.
        # This loss is only used to select max negatives.
        loss_conf_proxy = log_sum_exp(batch_conf) - batch_conf.gather(
            1, conf_t.view(-1, 1))
        loss_conf_proxy = loss_conf_proxy.view(num, -1)
        # Exclude positives
        loss_conf_proxy[pos] = 0
        # Exclude easy negatives
        ignore_neg_idx = ((conf_t <= 0) +
                          (arm_score[:, :, 1] < self.pos_prior_threshold)
                          ).gt(1)
        loss_conf_proxy[ignore_neg_idx] = 0
        # Sort and select max negatives
        # Values in loss_c are not less than 0.
        _, loss_idx = loss_conf_proxy.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # pdb.set_trace()
        num_pos = pos.long().sum(1, keepdim=True)
        max_neg = (loss_conf_proxy > 0).long().sum(1, keepdim=True)
        # print(max_neg)
        num_neg = torch.min(self.neg_pos_ratio * num_pos, max_neg)
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
