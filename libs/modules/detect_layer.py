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
        self.top_k = top_k
        self.pos_prior_threshold = pos_prior_threshold
        # Parameters used in nms.
        self.detect_conf_thresh = detect_conf_thresh
        self.nms_thresh = nms_thresh
        self.arm_variance = arm_variance
        self.variance = odm_variance
        # self.softmax = functional.softmax

    def forward(self, bi_predictions,multi_predictions, prior_data):
        """
        :param bi_predictions:
            0).arm_loc_data: (tensor) location predictions from loc layers of ARM
            Shape: (batch_size, num_priors, 4)
            1).arm_conf_data: (tensor) confidence predictions from conf layers of ARM
            Shape: (batch_size, num_priors, 2)
        :param multi_predictions:
            0).odm_loc_data: (tensor) location predictions from loc layers of ODM
            Shape: (batch_size, num_priors, 4)
            1).odm_conf_data: (tensor) confidence predictions from conf layers of ODM
            Shape: (batch_size, num_priors, num_classes)
        :param prior_data: (tensor) Prior boxes and from priorbox layers
            Shape: (num_priors, 4)
        """
        # batch size
        arm_loc_data, arm_conf_data = (bi_predictions[0].data,
                                     bi_predictions[1].data)
        loc_data, conf_data = (multi_predictions[0].data,
                                           multi_predictions[1].data)
        # change confidence value to score by softmax.
        
        arm_score_data = functional.softmax(
            Variable(arm_conf_data.clone(), requires_grad=False), dim=-1
        ).data
        
        score_data = functional.softmax(
            Variable(conf_data.clone(), requires_grad=False), dim=-1
        ).data
        # batch size
        num = arm_loc_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5).type_as(
            loc_data)
        
        refined_priors = refine_priors(arm_loc_data, prior_data, self.arm_variance)
        # select
        for i in range(num):
            cur_arm_score = arm_score_data[i]
            # ignore priors whose positive score is small
            flag = cur_arm_score[:, 1] > self.pos_prior_threshold
            index = torch.nonzero(flag)[:, 0]
            # decoded boxes
            cur_refined_priors = refined_priors[i]
            all_boxes = decode(odm_loc_data[i], cur_refined_priors,
                               self.variance)
            odm_boxes = all_boxes[index]
            # odm_boxes = decode(odm_loc_data[i][index, :],
            #                    cur_refined_priors[index], self.variance)
            cur_odm_score = odm_score_data[i][index].clone().\
              transpose(1, 0)
            
            # pdb.set_trace()
            for cl in range(1, self.num_classes):
                c_mask = cur_odm_score[cl].gt(self.detect_conf_thresh)
                # pdb.set_trace()
                # print(type(c_mask))
                scores = cur_odm_score[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(odm_boxes)
                boxes = odm_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        # sort across each batch, which is different from the paper.
        # But since fill_ function is used, this is useless.
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        
        return output
