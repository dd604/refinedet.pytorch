# -*- coding: utf-8 -*-
# --------------------------------------------------------
# RefineDet in PyTorch
# Written by Dongdong Wang
# Official and original Caffe implementation is at
# https://github.com/sfzhang15/RefineDet
# --------------------------------------------------------

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from libs.utils.box_utils import decode, nms
import pdb
sys.dont_write_bytecode = True

class Detect(nn.Module):
    """At test time, Detect is the final layer of RefineDet.
    Decode location preds, apply non-maximum suppression to location predictions
    based on conf scores and threshold to a top_k number of output predictions
    for both confidence score and locations.
    """
    def __init__(self, num_classes, odm_variance,
                 top_k_pre_class, top_k, detect_conf_thresh, nms_thresh):
        """
        :param num_classes: number of classes.
        :param variance:
        :param top_k_pre_class: keep the top k for nms in each class.
        :param top_k: keep the top k of detection results.
        :param detect_conf_thresh: keep detections whoes confidence is big.
        :param nms_thresh:
        """
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.top_k_per_class = top_k_pre_class
        self.keep_top_k = top_k

        # Parameters used in nms.
        self.detect_conf_thresh = detect_conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = odm_variance


    def forward(self, odm_predictions, refined_anchors,
                ignore_flags_refined_anchor):
        """
        :param odm_predictions:
            0).odm_loc_data: (tensor) location predictions from loc layers of ODM
            Shape: (batch_size, num_anchors, 4)
            1).odm_conf_data: (tensor) confidence predictions from conf layers of ODM
            Shape: (batch_size, num_anchors, num_classes)
        :param refined_anchors: (batch_size, num_anchors, 4)
        :param ignore_flags_refined_anchor: (batch_size, num_anchors),
            1 means an igored negative anchor, otherwise reserved.
            
        """
        # pdb.set_trace()
        loc_data = odm_predictions[0].data
        score_data = functional.softmax(odm_predictions[1].detach(),
                                        dim=-1).data
        # Output
        num = refined_anchors.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k_per_class,
                             5).type_as(loc_data)
        # select
        # For each image, keep keep_top_k,
        # retain top_k per class for nms.
        for idx in range(num):
            # Decoded odm bbox prediction to get boxes
            all_boxes = decode(loc_data[idx], refined_anchors[idx],
                               self.variance)
            # Ignore predictions whose positive scores are small.
            # pdb.set_trace()
            flag = ignore_flags_refined_anchor[idx].data < 1
            box_flag = flag.unsqueeze(flag.dim()).expand_as(all_boxes)
            conf_flag = flag.unsqueeze(flag.dim()).expand_as(score_data[idx])
            select_boxes = all_boxes[box_flag].view(-1, 4)
            # ?
            select_scores = score_data[idx][conf_flag].view(
                -1, self.num_classes).transpose(1, 0)
            # NMS per class
            for icl in range(1, self.num_classes):
                c_mask = select_scores[icl].gt(self.detect_conf_thresh)
                # pdb.set_trace()
                # print(type(c_mask))
                scores = select_scores[icl][c_mask]
                if len(scores) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(select_boxes)
                boxes = select_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh,
                                 self.top_k_per_class)
                output[idx, icl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        # Sort each image,
        # But since fill_ function is used, this is useless.
        # pdb.set_trace()
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        
        return flt.view(num, self.num_classes, self.top_k_per_class, 5)
