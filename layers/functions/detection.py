import torch
from torch.autograd import Function, Variable
import torch.nn.functional as functional
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, prior_threshold,
                 conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.prior_threshold = prior_threshold
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, bi_loc_data, bi_conf_data, multi_loc_data,
                multi_conf_data, prior_data):
        """
        Args:
            bi_loc_data: (tensor) binary location preds from loc layers
                Shape: [batch,num_priors*4]
            bi_conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        # import pdb
        # pdb.set_trace()
        num = bi_loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        bi_conf_preds = bi_conf_data.view(num, num_priors, 2)
        multi_conf_preds = multi_conf_data.view(num, num_priors,
                                                self.num_classes)
        bi_conf_preds_variable = Variable(bi_conf_preds, requires_grad=True)
        # select
        # Decode predictions into bboxes.
        for i in range(num):
            # For each class, perform nms
            # softmax
            # import pdb
            # pdb.set_trace()
            # print(type(bi_conf_preds_variable))
            bi_conf_scores = functional.softmax(bi_conf_preds_variable[i],
                                                dim=-1).data.clone()
            ignore_flag = bi_conf_scores[:, 0] > self.prior_threshold
            index = torch.nonzero(1 - ignore_flag)[:, 0]
            # decoded boxes
            arm_boxes = decode(bi_loc_data[i][index, :],
                               prior_data[index, :], self.variance)
            odm_boxes = decode(multi_loc_data[i][index, :],
                               arm_boxes, self.variance)
            multi_conf_scores = multi_conf_preds[i][index, :].clone().\
              transpose(1, 0)
            
            # pdb.set_trace()
            for cl in range(1, self.num_classes):
                c_mask = multi_conf_scores[cl].gt(self.conf_thresh)
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
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
