# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp, encode, decode, match_and_encode

import pdb


class BiBoxLoss(nn.Module):
  """SSD Weighted Loss Function
  Compute Targets:
    1) Produce Confidence Target Indices by matching  ground truth boxes
       with (default) 'priorboxes' that have jaccard index > threshold parameter
       (default threshold: 0.5).
    2) Produce localization target by 'encoding' variance into offsets of ground
       truth boxes and their matched  'priorboxes'.
    3) Hard negative mining to filter the excessive number of negative examples
       that comes with using a large number of default bounding boxes.
       (default negative:positive ratio 3:1)
  Objective Loss:
    L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
    weighted by α which is set to 1 by cross val.
    Args:
        c: class confidences,
        l: predicted boxes,
        g: ground truth boxes
        N: number of matched default boxes
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
  """

  def __init__(self, overlap_thresh, prior_for_matching, bkg_label,
               neg_mining, neg_pos, neg_overlap,
               use_gpu=True):
    super(BiBoxLoss, self).__init__()
    self.use_gpu = use_gpu
    self.threshold = overlap_thresh
    self.background_label = bkg_label
    self.use_prior_for_matching = prior_for_matching
    self.num_classes = 2
    self.do_neg_mining = neg_mining
    self.negpos_ratio = neg_pos
    self.neg_overlap = neg_overlap
    self.variance = cfg['variance']

  def forward(self, predictions, priors, targets):
    """Binary box and classification Loss
    Args:
        predictions (tuple): A tuple containing loc preds, conf preds,
        and priors boxes from SSD net.
            conf shape: torch.size(batch_size,num_priors,num_classes)
            loc shape: torch.size(batch_size,num_priors,4)
            priors shape: torch.size(num_priors,4)

        targets (tensor): Ground truth boxes and labels for a batch,
            shape: [batch_size,num_objs,5] (last idx is the label).
    """
    # pdb.set_trace()
    loc_data, conf_data = predictions
    num = loc_data.size(0)
    priors = priors[:loc_data.size(1), :]
    num_priors = (priors.size(0))
    
    # result: match priors (default boxes) and ground truth boxes
    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)
    for idx in range(num):
      truths = targets[idx][:, :-1].data
      # labels = targets[idx][:, -1].data.clone()
      # labels.fill_(0)
      labels = targets[idx][:, -1].data
      # binary classes
      # labels = (labels >= 0).type_as(labels)
      # labels = labels.fill(0)
      
      labels = labels.new(labels.size()).zero_()
      
      defaults = priors.data
      # pdb.set_trace()
      match(self.threshold, truths, defaults, self.variance, labels,
            loc_t, conf_t, idx)
      
    
    if self.use_gpu:
      loc_t = loc_t.cuda()
      conf_t = conf_t.cuda()
    # wrap targets
    loc_t = Variable(loc_t, requires_grad=False)
    conf_t = Variable(conf_t, requires_grad=False)

    pos = conf_t > 0
    # pdb.set_trace()
    # Localization Loss (Smooth L1)
    # Shape: [batch,num_priors,4]
    pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
    loc_p = loc_data[pos_idx].view(-1, 4)
    loc_t = loc_t[pos_idx].view(-1, 4)
    loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

    # Compute max conf across batch for hard negative mining
    batch_conf = conf_data.view(-1, self.num_classes)
    loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

    # change view
    loss_c = loss_c.view(num, -1)
    # Hard Negative Mining
    loss_c[pos] = 0  # filter out pos boxes for now
    
    _, loss_idx = loss_c.sort(1, descending=True)
    # pdb.set_trace()
    _, idx_rank = loss_idx.sort(1)
    num_pos = pos.long().sum(1, keepdim=True)
    num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
    neg = idx_rank < num_neg.expand_as(idx_rank)

    # Confidence Loss Including Positive and Negative Examples
    pos_idx = pos.unsqueeze(2).expand_as(conf_data)
    neg_idx = neg.unsqueeze(2).expand_as(conf_data)
    conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
    targets_weighted = conf_t[(pos+neg).gt(0)]
    loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

    # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

    N = num_pos.data.sum()
    loss_l /= N
    loss_c /= N

    return loss_l, loss_c
  


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap,
                 priors_refine_threshold,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.use_prior_for_matching = prior_for_matching
        self.priors_refine_threshold = priors_refine_threshold
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.arm_variance = cfg['variance']

    def forward(self, bi_prediction, multi_prediction, priors, targets):
        """Multibox Loss
        Args:
            bi_prediction (tuple): A tuple containing loc preds, conf preds,
            multi_prediction (tuple): A tuple containing loc preds, conf preds,
            and priors boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # tensor
        arm_loc_data = bi_prediction[0].data
        # no soft max score
        arm_conf_data = bi_prediction[1].data
        # arm_conf_data = F.softmax(bi_prediction[1], -1).data
        # variable
        loc_data, conf_data = multi_prediction
        num = loc_data.size(0)
        num_priors = priors.size(0)
      
        # adjust priors
        refined_priors = self._adjust_priors(arm_loc_data, priors.data,
                                             self.arm_variance)
        # match priors (default boxes) and ground truth boxes
        # consider each image in one batch.
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors).fill_(-1)
        
        tmp_loc_t = torch.Tensor(num_priors, 4)
        tmp_conf_t = torch.Tensor(num_priors)
        # pdb.set_trace()
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            # select priors.
            # use confidence data of arm
            cur_priors = refined_priors[idx]
            # softmax arm_conf_data[idx].
            arm_negative_scores = arm_conf_data[idx, :, 0]
            
            ignore_flag = arm_negative_scores > self.priors_refine_threshold
            index = torch.nonzero(1 - ignore_flag)[:, 0]
            # print(index.size())
            used_priors = cur_priors[index, :]
            # used_priors = cur_priors.index_select(0, index)
            used_loc_t, used_conf_t = match_and_encode(self.threshold, truths,
                                                       used_priors, self.variance,
                                                       labels)
            # unmap
            tmp_loc_t.fill_(0)
            tmp_conf_t.fill_(-1)
            tmp_loc_t[index, :] = used_loc_t
            tmp_conf_t[index] = used_conf_t
            loc_t[idx, :, :] = tmp_loc_t
            conf_t[idx, :] = tmp_conf_t
            
            # loc_t[idx, index, :] = used_loc_t
            # conf_t[idx, index] = used_conf_t
      
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        
        # positives
        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # use all gt priors for location loss.
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # conf_t must be in range [0, classes]
        ignore = conf_t == -1
        conf_t[ignore] = 0
        # pdb.set_trace()
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        
        # (num, num_priors)
        loss_c = loss_c.view(num, -1)
        # Hard Negative Mining, only considering classification scores
        loss_c[pos] = 0  # filter out pos boxes for now
        # filter out ignore boxes
        loss_c[ignore] = 0
        # negatives
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        # num_neg <= num_pos
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # 1 for positives
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        
        return loss_l, loss_c


    def _adjust_priors(self, arm_loc_data, priors, variance):
      num = arm_loc_data.size(0)
      num_priors = priors.size(0)
      
      assert arm_loc_data.size(1) == num_priors, 'priors'
      
      refined_priors = torch.Tensor(num, num_priors, 4)
      for ind in range(num):
        cur_loc = arm_loc_data[ind, :, :]
        refined_priors[ind] = decode(cur_loc, priors, variance)
      
      return refined_priors
