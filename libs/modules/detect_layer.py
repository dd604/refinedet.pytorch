import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from libs.utils.box_utils import decode, nms, refine_priors

sys.dont_write_bytecode = True

class Detect(nn.Module):
    """At test time, Detect is the final layer of RefineDet.
    Decode location preds, apply non-maximum suppression to location predictions
    based on conf scores and threshold to a top_k number of output predictions
    for both confidence score and locations.
    """
    def __init__(self, num_classes, top_k, pos_prior_threshold,
                 detect_conf_thresh, nms_thresh, arm_variance,
                 odm_variance):
        """
        :param num_classes: number of classes.
        :param top_k: keep the top k of detection results.
        :param pos_prior_threshold: filter priors, tipically 0.99. For a prior whoes positive
            probability is less than pos_prior_threshold will be missed.
        :param detect_conf_thresh: keep detections whoes confidence is big.
        :param nms_thresh:
        """
        self.num_classes = num_classes
        self.top_k = top_k
        self.pos_prior_threshold = pos_prior_threshold
        # Parameters used in nms.
        self.detect_conf_thresh = detect_conf_thresh
        self.nms_thresh = nms_thresh
        self.arm_variance = arm_variance
        self.variance = odm_variance

    def forward(self, bi_predictions, multi_predictions, prior_data):
        """
        :param bi_predictions:
            0).bi_loc_data: (tensor) location predictions from loc layers of ARM
            Shape: [batch, num_priors*4]
            1).bi_conf_data: (tensor) confidence predictions from conf layers of ARM
            Shape: [batch*num_priors, 2]
        :param multi_predictions:
            0).multi_loc_data: (tensor) location predictions from loc layers of ODM
            Shape: [batch, num_priors*4]
            1).multi_conf_data: (tensor) confidence predictions from conf layers of ODM
            Shape: [batch*num_priors, num_classes]
        :param prior_data: (tensor) Prior boxes and from priorbox layers
            Shape: [num_priors, 4]
        """
        # import pdb
        # pdb.set_trace()
        # batch size
        bi_loc_data, bi_conf_data = bi_predictions[0], bi_predictions[1]
        multi_loc_data, multi_conf_data = multi_predictions[0], \
                                          multi_predictions[1]
        num = bi_loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        bi_conf_preds = bi_conf_data.view(num, num_priors, 2)
        multi_conf_preds = multi_conf_data.view(num, num_priors,
                                                self.num_classes)
        bi_conf_preds_variable = Variable(bi_conf_preds, requires_grad=False)
        refined_priors = refine_priors(bi_loc_data, prior_data, self.arm_variance)
        # select
        # Decode predictions into bboxes.
        for i in range(num):
            # For each class, perform nms
            # softmax
#             pdb.set_trace()
            # print(type(bi_conf_preds_variable))
            bi_conf_scores = functional.softmax(bi_conf_preds_variable[i],
                                                dim=-1).data.clone()
            # ignore priors whose positive score is small
            flag = bi_conf_scores[:, 1] >= self.pos_prior_threshold
            index = torch.nonzero(flag)[:, 0]
            # decoded boxes
            cur_refined_priors = refined_priors[i]
            odm_boxes = decode(multi_loc_data[i][index, :],
                               cur_refined_priors[index], self.variance)
            multi_conf_scores = multi_conf_preds[i][index, :].clone().\
              transpose(1, 0)
            
            # pdb.set_trace()
            for cl in range(1, self.num_classes):
                c_mask = multi_conf_scores[cl].gt(self.detect_conf_thresh)
                scores = multi_conf_scores[cl][c_mask]
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
