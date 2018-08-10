import torch
from torch.autograd import Function, Variable
import torch.nn.functional as functional
from ..box_utils import decode, nms
from data import voc as cfg
import pdb


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, prior_threshold,
                 conf_thresh, nms_thresh):
        """
        prior_threshold: to filter negatives, tipically 0.99
        conf_thresh: for results
        """
        # self.num_classes = num_classes
        self.num_classes = 2
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
        pdb.set_trace()
        num = bi_loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        bi_conf_preds = bi_conf_data.view(num, num_priors, 2)
        bi_conf_preds_variable = Variable(bi_conf_preds, requires_grad=False)
        # select
        # Decode predictions into bboxes.
        for i in range(num):
            # For each class, perform nms
#             pdb.set_trace()
            bi_conf_scores = functional.softmax(bi_conf_preds_variable[i],
                                                dim=-1).transpose(1, 0).data.clone()
            pdb.set_trace()
            odm_boxes = decode(bi_loc_data[i],
                   prior_data, self.variance)
            
            for cl in range(1, self.num_classes):
            # for cl in range(0, 1):
                c_mask = bi_conf_scores[cl].gt(self.conf_thresh)
                scores = bi_conf_scores[cl][c_mask]
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
