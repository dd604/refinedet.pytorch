# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from libs.utils.box_utils import log_sum_exp, match_with_flags

import pdb

class ODMLoss(nn.Module):
    """
    """
    
    def __init__(self, num_classes, overlap_thresh,
                 neg_pos_ratio, variance):
        super(ODMLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance
    
    def forward(self, odm_predictions, refined_anchors,
                ignore_flags_refined_anchor, targets):
        """Multibox Loss
        :param odm_predictions: tuple (odm_loc_pred, odm_conf_pred),
            odm_loc_pred (batch_size, num_anchors, 4),
            odm_conf_pred (batch_size, num_anchors, num_classes)
        :param refined_anchors: (batch_size, num_anchors, 4)
        :param ignore_flags_refined_anchor: (batch_size, num_anchors),
        1 means an igored negative anchor, otherwise reserved.
        :param targets: Ground truth boxes and labels for a batch,
        shape, [batch_size,num_objs,5]
        (last idx is the label, 0 for background, >0 for target).
        """
        (loc_pred, conf_pred) = odm_predictions
        num = refined_anchors.size(0)
        num_anchors = refined_anchors.size(1)
        # loc_t = torch.Tensor(num, num_anchors, 4)
        # conf_t = torch.LongTensor(num, num_anchors)
        # if loc_pred.is_cuda:
        #     loc_t = loc_t.cuda()
        #     conf_t = conf_t.cuda()
        loc_t = loc_pred.data.new(num, num_anchors, 4)
        conf_t = loc_pred.data.new(num, num_anchors).long()
        
        # Match refined_anchors (predicted ROIs) and ground truth boxes
        # Consider each image in one batch.
#         pdb.set_trace()
        for idx in range(num):
            cur_targets = targets[idx].data
            valid_targets = cur_targets[cur_targets[:, -1] > 0]
            gt_boxes = valid_targets[:, :-1]
            labels = valid_targets[:, -1]
            
            # truths = targets[idx][:, :-1].data
            # labels = targets[idx][:, -1].data
            # Refined anchors of this idx
            cur_anchors = refined_anchors[idx]
            #pdb.set_trace()
            cur_ignore_flags = ignore_flags_refined_anchor[idx].data
            # valid anchors.
            match_with_flags(self.overlap_thresh, gt_boxes, cur_anchors,
                             cur_ignore_flags, self.variance, labels,
                             loc_t, conf_t, idx)
        
        # Wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        # valid indice.
        pos_loc_flag = loc_t.new(loc_t).byte()
        pos_conf_flag = conf_t.new(conf_t).byte()
        neg_conf_flag = conf_t.new(conf_t).byte()
        # pdb.set_trace()

        for idx in xrange(num):
            single_conf_t = conf_t[idx]
            pos = single_conf_t > 0
            pos_loc_flag[idx] = pos.unsqueeze(1).expand_as(loc_t[idx])
    
            # Mimic MAX_NEGATIVE of caffe-ssd
            # Compute max conf across a batch for selecting negatives with large
            # error confidence.
            single_conf_pred = conf_pred[idx]
            # Sum up losses of all wrong classes.
            # This loss is only used to select max negatives.
            loss_conf_proxy = (log_sum_exp(single_conf_pred) - single_conf_pred.gather(
                1, single_conf_t.view(-1, 1))).view(-1)
            # Exclude positives
            loss_conf_proxy[pos] = 0
            # Exclude ingored refined anchors.
            loss_conf_proxy[ignore_flags_refined_anchor[idx].gt(0)] = 0
            # Sort and select max negatives
            # Values in loss_c are not less than 0.
            _, loss_idx = loss_conf_proxy.sort(0, descending=True)
            _, idx_rank = loss_idx.sort(0)
            # pdb.set_trace()
            num_pos = torch.sum(pos)
            max_neg = torch.sum(loss_conf_proxy > 0)
            # print(max_neg)
            num_neg = torch.min(self.neg_pos_ratio * num_pos, max_neg)
            neg = idx_rank < num_neg.expand_as(idx_rank)
            # Total confidence loss includes positives and negatives.
            pos_conf_flag[idx] = pos
            neg_conf_flag[idx] = neg

        pos_loc_t = loc_t[pos_loc_flag.detach()].view(-1, 4)
        # # Select postives to compute bounding box loss.
        pos_loc_pred = loc_pred[pos_loc_flag.detach()].view(-1, 4)
        # loss_l = functional.smooth_l1_loss(pos_loc_pred, pos_loc_t, size_average=False)
        loss_l = functional.smooth_l1_loss(pos_loc_pred, pos_loc_t, reduction='sum')
        # pdb.set_trace()
        # Final classification loss
        conf_keep = (pos_conf_flag + neg_conf_flag).view(-1).gt(0).nonzero().view(-1)
        valid_conf_pred = torch.index_select(conf_pred.view(-1, self.num_classes), 0, conf_keep)
        valid_conf_t = torch.index_select(conf_t.view(-1), 0, conf_keep)
        # loss_c = functional.cross_entropy(valid_conf_pred, valid_conf_t, size_average=False)
        loss_c = functional.cross_entropy(valid_conf_pred, valid_conf_t, reduction='sum')
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + alpha*Lloc(x,l,g)) / N
        # only number of positives
        total_num = torch.sum(pos_conf_flag).float()
        # print('arm_loss', loss_l, loss_c, total_num)
        # pdb.set_trace()
        loss_l /= total_num
        loss_c /= total_num

        return loss_l, loss_c
