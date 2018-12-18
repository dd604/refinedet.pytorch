import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from libs.utils.box_utils import decode, nms, refine_priors
import pdb
sys.dont_write_bytecode = True

class Detect(nn.Module):
    """At test time, Detect is the final layer of RefineDet.
    Decode location preds, apply non-maximum suppression to location predictions
    based on conf scores and threshold to a top_k number of output predictions
    for both confidence score and locations.
    """
    def __init__(self, num_classes, arm_variance, odm_variance,
                 top_k, pos_prior_threshold, detect_conf_thresh, nms_thresh):
        """
        :param num_classes: number of classes.
        :param arm_variance:
        :param odm_variance:
        :param top_k: keep the top k of detection results.
        :param pos_prior_threshold: filter priors, tipically 0.01. For a prior whoes positive
            score (probability) is less than pos_prior_threshold will be missed.
        :param detect_conf_thresh: keep detections whoes confidence is big.
        :param nms_thresh:
        """
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.top_k_per_class = 400
        self.keep_top_k = 200

        self.pos_prior_threshold = pos_prior_threshold
        # Parameters used in nms.
        self.detect_conf_thresh = detect_conf_thresh
        self.nms_thresh = nms_thresh
        self.arm_variance = arm_variance
        self.variance = odm_variance
        # self.softmax = functional.softmax

    def forward(self, arm_predictions, odm_predictions, prior_data):
        """
        :param arm_predictions:
            0).arm_loc_data: (tensor) location predictions from loc layers of ARM
            Shape: (batch_size, num_priors, 4)
            1).arm_conf_data: (tensor) confidence predictions from conf layers of ARM
            Shape: (batch_size, num_priors, 2)
        :param odm_predictions:
            0).odm_loc_data: (tensor) location predictions from loc layers of ODM
            Shape: (batch_size, num_priors, 4)
            1).odm_conf_data: (tensor) confidence predictions from conf layers of ODM
            Shape: (batch_size, num_priors, num_classes)
        :param prior_data: (tensor) Prior boxes and from priorbox layers
            Shape: (num_priors, 4)
        """
        # Compute prediction scores using softmax for tow modules.
        arm_loc_data = arm_predictions[0].data
        arm_score_data = functional.softmax(arm_predictions[1].detach(),
                                            dim=-1).data
        loc_data = odm_predictions[0].data
        score_data = functional.softmax(odm_predictions[1].detach(),
                                        dim=-1).data
        # Output
        num = arm_loc_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k_per_class,
                             5).type_as(loc_data)
        # Predict ROIs of ARM and convert them to priors.
        refined_priors = refine_priors(arm_loc_data, prior_data, self.arm_variance)
        # select
        # For each image, keep keep_top_k,
        # retain top_k per class for nms.
        for idx in range(num):
            # Decoded odm bbox prediction to get boxes
            all_boxes = decode(loc_data[idx], refined_priors[idx],
                               self.variance)
            cur_arm_score = arm_score_data[idx]
            # Ignore predictions whose positive scores are small.
            flag = cur_arm_score[:, 1] > self.pos_prior_threshold
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
                if scores.dim() == 0:
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
        # flt_copy = flt.clone()
        # flt[(rank > self.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        # print(torch.sum(flt_copy != flt))
        #flt[(rank > self.keep_top_k).unsqueeze(-1).expand_as(flt)] = 0
        # print(torch.sum(flt_copy != flt))
        # flt.view(num, self.num_classes, self.top_k_per_class, 5)
        
        return flt.view(num, self.num_classes, self.top_k_per_class, 5)
