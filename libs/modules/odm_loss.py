# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from libs.utils.box_utils import log_sum_exp, match_and_encode, refine_priors


class ODMLoss(nn.Module):
    """
    """
    
    def __init__(self, num_classes, pos_prior_threshold, overlap_thresh,
                 neg_pos_ratio, arm_variance, variance):
        super(ODMLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.pos_prior_threshold = pos_prior_threshold
        self.negpos_ratio = neg_pos_ratio
        self.variance = variance
        self.arm_variance = arm_variance
    
    def forward(self, bi_prediction, multi_prediction, priors, targets):
        """Multibox Loss
        """
        # import pdb
        # pdb.set_trace()
        
        arm_loc_data = bi_prediction[0].data
        # softmax
        # arm_conf_data = functional.softmax(bi_prediction[1].data, -1)
        arm_prob_data = functional.softmax(bi_prediction[1], -1).data
        # variable
        odm_loc_data, odm_conf_data = multi_prediction[0], multi_prediction[1]
        num = odm_loc_data.size(0)
        num_priors = priors.size(0)
        
        # adjust priors
        # the shape is [num, num_priors, 4], priors [num_priors, 4]
        refined_priors = refine_priors(arm_loc_data, priors.data,
                                       self.arm_variance)
        # match priors (default boxes) and ground truth boxes
        # consider each image in one batch.
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors).fill_(-1)
        tmp_loc_t = torch.Tensor(num_priors, 4)
        tmp_conf_t = torch.LongTensor(num_priors)
        if arm_loc_data.is_cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            tmp_loc_t = tmp_loc_t.cuda()
            tmp_conf_t = tmp_conf_t.cuda()
        
        # pdb.set_trace()
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            # select priors.
            # use confidence data of arm
            cur_priors = refined_priors[idx]
            # softmax arm_conf_data[idx].
            arm_positive_probs = arm_prob_data[idx, :, 1]
            reserve_flag = arm_positive_probs > self.pos_prior_threshold
            index = torch.nonzero(reserve_flag)[:, 0]
            used_priors = cur_priors[index]
            # print(used_priors.shape)
            # the type of used_conf_t is the same as lables.
            # need to convert it to LongTensor
            used_loc_t, used_conf_t = match_and_encode(self.threshold, truths,
                               used_priors, self.variance, labels)
            # unmap
            tmp_loc_t.fill_(0)
            tmp_conf_t.fill_(-1)
            tmp_loc_t[index, :] = used_loc_t
            # convert used_conf_t to LongTensor
            tmp_conf_t[index] = used_conf_t.long()
            loc_t[idx] = tmp_loc_t
            conf_t[idx] = tmp_conf_t
        
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        
        # positives
        pos = conf_t > 0
        
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # use all gt priors for location loss.
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(odm_loc_data)
        loc_pred = odm_loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = functional.smooth_l1_loss(loc_pred, loc_t, size_average=False)
        
        # Compute max conf across batch for hard negative mining
        batch_conf = odm_conf_data.view(-1, self.num_classes)
        # conf_t must be long integers with range [0, classes]
        ignore = (conf_t == -1)
        # print([ignore[i].long().sum() for i in xrange(num)])
        # these ignored examples will not be selected, since select with loss_c
        # in case of some is selected
        conf_t[ignore] = 0
        # log(sum(exp(batch_conf)))
        # conf_t.view(-1, 1) as index, this loss is only used for sort
        # and negative sampling, is not the final classification loss.
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        
        # (num, num_priors)
        loss_c = loss_c.view(num, -1)
        # Hard Negative Mining, only considering classification scores
        # filter out pos boxes
        loss_c[pos] = 0
        # filter out ignore boxes
        loss_c[ignore] = 0
        # values in loss_c are bigger than 0.
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        # num_neg = torch.clamp(self.negpos_ratio * num_pos, max=(pos.size(1) - num_pos))
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # print(num_pos)
        # print(num_neg)
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(odm_conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(odm_conf_data)
        conf_pred = odm_conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # 1 for positives
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # final classification loss
        loss_c = functional.cross_entropy(conf_pred, targets_weighted, size_average=False)
        
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + alpha*Lloc(x,l,g)) / N
        
        total_num = num_pos.data.sum()
        loss_l /= total_num
        loss_c /= total_num
        
        return loss_l, loss_c
