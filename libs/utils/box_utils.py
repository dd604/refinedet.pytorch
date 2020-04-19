# -*- coding: utf-8 -*-
import torch
import pdb


def point_form(boxes):
    """ Convert anchor_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from anchorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert anchor_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                      boxes[:, 2:] - boxes[:, :2]), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(gt_boxes, anchors):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        gt_boxes: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        anchors: (tensor) Prior boxes from anchorbox layers, Shape: [num_anchors,4]
    Return:
        jaccard overlap: (tensor) Shape: [gt_boxes.size(0), anchors.size(0)]
    """
    # pdb.set_trace()
    k = gt_boxes.size(0)
    n = anchors.size(0)
    # expand dims (k, n, 4)
    ex_gt_boxes = gt_boxes.view(k, 1, 4).expand(k, n, 4)
    ex_anchors = anchors.view(1, n, 4).expand(k, n, 4)
    iw = (torch.min(ex_gt_boxes[:, :, 2], ex_anchors[:, :, 2]) -
          torch.max(ex_gt_boxes[:, :, 0], ex_anchors[:, :, 0]))
    iw[iw < 0] = 0
    ih = (torch.min(ex_gt_boxes[:, :, 3], ex_anchors[:, :, 3]) -
          torch.max(ex_gt_boxes[:, :, 1], ex_anchors[:, :, 1]))
    ih[ih < 0] = 0
    
    EPS = 1e-5
    gt_boxes_x = ex_gt_boxes[:, :, 2] - ex_gt_boxes[:, :, 0]
    gt_boxes_y = ex_gt_boxes[:, :, 3] - ex_gt_boxes[:, :, 1]
    # [k, n]
    gt_boxes_area = gt_boxes_x * gt_boxes_y
    # zero gts.
    gt_boxes_zero = (torch.abs(gt_boxes_x) < EPS) & (torch.abs(gt_boxes_y) < EPS)
    anchors_x = ex_anchors[:, :, 2] - ex_anchors[:, :, 0]
    anchors_y = ex_anchors[:, :, 3] - ex_anchors[:, :, 1]
    # [num_gts, num_anchors]
    anchors_area = anchors_x * anchors_y
    # anchors_zero = (torch.abs(anchors_x) < EPS) & (torch.abs(anchors_y) < EPS)
    anchors_zero = (anchors_x < EPS) & (anchors_y < EPS)
    inner = iw * ih
    overlaps = inner / (gt_boxes_area + anchors_area - inner + EPS)
    
    overlaps.masked_fill_(gt_boxes_zero, 0)
    overlaps.masked_fill_(anchors_zero, -1)
    
    return overlaps

def match(threshold, truths, anchors, variances, labels, loc_t, conf_t, idx):
    """Match each anchor box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_anchors].
        anchors: (tensor) Prior boxes from anchorbox layers, Shape: [n_anchors,4].
        variances: (tensor) Variances corresponding to each anchor coord,
            Shape: [num_anchors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        indice is from 0.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(anchors)
    )
    # (Bipartite Matching)
    # [1,num_gts] best anchor for each ground truth
    best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)
    # best_anchor_overlap[best_anchor_overlap == 0] = 1e-5
    # pdb.set_trace()
    # [1,num_anchors] best ground truth for each anchor
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_anchor_idx.squeeze_(1)
    best_anchor_overlap.squeeze_(1)
    # ensure that gts must being matched.
    # rise overlap
    best_anchor_idx = best_anchor_idx[best_anchor_overlap > 0]
    if len(best_anchor_idx) > 0:
        best_truth_overlap.index_fill_(0, best_anchor_idx, 1.0)
    # For each gt, ensure it matches with its anchor of max overlap
    for j in range(best_anchor_idx.size(0)):
        best_truth_idx[best_anchor_idx[j]] = j
    # Select
    matches = truths[best_truth_idx]  # Shape: [num_anchors,4]

    conf = labels[best_truth_idx]  # Shape: [num_anchors]
    # conf = labels[best_truth_idx] + 1  # Shape: [num_anchors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    # All location is stored ? but use conf > 0 to indicate which is valid.
    # It is better to modify loc to zeros, or weights to indicate
    # like faster rcnn
    loc = encode(matches, anchors, variances)
    loc_t[idx] = loc  # [num_anchors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_anchors] top class label for each anchor


def match_with_flags(threshold, truths, anchors, ignore_flags, variances,
                     labels, loc_t, conf_t, idx):
    """Match each anchor box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_anchors].
        anchors: (tensor) Prior boxes from anchorbox layers, Shape: [n_anchors,4].
        ignore_flags: (tensor) Flags indicate if to ignore this anchor or not. [n_anchors]
        variances: (tensor) Variances corresponding to each anchor coord,
            Shape: [num_anchors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        indice is from 0.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # pdb.set_trace()
    # jaccard index
    # After refinement, some anchors may have zero width or height, causing overlaps to be Nan.
    overlaps = jaccard(
        truths,
        point_form(anchors)
    )   
    # (Bipartite Matching)
    # [1,num_objects] best anchor for each ground truth
    best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)
    # [1,num_anchors] best ground truth for each anchor
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_anchor_idx.squeeze_(1)
    best_anchor_overlap.squeeze_(1)
#     pdb.set_trace()
    # Ensure anchors matched with selected gts have big overlap.
    best_anchor_idx = best_anchor_idx[best_anchor_overlap > 0]
    if len(best_anchor_idx) > 0:
        best_truth_overlap.index_fill_(0, best_anchor_idx, 2)
    # Since we use best_truth_overlap to get flags, we do not need to modify best_truth_idx.
    # For each selected gt, ensure every gt matches with its anchor of max overlap
    for j in range(best_anchor_idx.size(0)):
        best_truth_idx[best_anchor_idx[j]] = j
    # Select
    matches = truths[best_truth_idx]  # Shape: [num_anchors,4]
    conf = labels[best_truth_idx] # Shape: [num_anchors]   
    # Since some elements in best_truth_overlap may be Nan, we use >= threshold
    background_flag = 1 - (best_truth_overlap >= threshold)
    conf[background_flag] = 0
    # All location is stored ? but use conf > 0 to indicate which is valid.
    # It is better to modify loc to zeros, or weights to indicate
    loc = encode(matches, anchors, variances)
    loc[background_flag.unsqueeze(-1).expand_as(loc)] = -1.
    loc_t[idx] = loc  # [num_anchors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_anchors] top class label for each anchor
    
def encode(matched, anchors, variances):
    """Encode the variances from the anchorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the anchor boxes.
    Args:
        matched: (tensor) Coords of ground truth for each anchor in point-form
            Shape: [num_anchors, 4]. (xmin, ymin, xmax, ymax)
        anchors: (tensor) Prior boxes in center-offset form
            Shape: [num_anchors,4]. (center_x, center_y, width, height)
        variances: (list[float]) Variances of anchorboxes
    Return:
        encoded boxes (tensor), Shape: [num_anchors, 4]
    """
    
    # dist b/t match center and anchor's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - anchors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * anchors[:, 2:])
    # match wh / anchor wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / (anchors[:, 2:] + 1e-10)
    # g_wh = torch.log(g_wh) / variances[1]
    g_wh = torch.log(g_wh + 1e-10) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_anchors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, anchors, variances):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_anchors,4]
        anchors (tensor): Prior boxes in center-offset form.
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchorboxes
    Return:
        decoded bounding box predictions
        (xmin, ymin, xmax, ymax)
    """
    
    boxes = torch.cat((
        anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    # pdb.set_trace()
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_anchors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_anchors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_anchors.
    """
    
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    
    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def refine_anchors(loc_pred, anchors, variance):
    """
    Refine location of anchors with location predicts.
    :param loc_pred: (batch_size, num_anchors, 4),
        (norm_cx, norm_cy, norm_w, norm_h)
    :param anchors: (num_anchors, 4), (cx, cy, w, h)
    :param variance: (var_xy, var_wh)
    :return: refined_anchors (batch_size, num_anchors, 4),
        (cx, cy, w, h)
    """
    num = loc_pred.size(0)
    num_anchors = anchors.size(0)
    
    assert loc_pred.size(1) == num_anchors, 'anchors'
    
    refined_anchors = torch.Tensor(num, num_anchors, 4).type_as(anchors)
    for ind in range(num):
        cur_loc = loc_pred[ind]
        # cur_loc (norm_dx, norm_dy, norm_w, norm_h)
        # anchors(cx, cy, w, h)
        # boxes (x1, y1, x2, y2)
        boxes = decode(cur_loc, anchors, variance)
        ori_boxes = boxes.clone()
        # (cx, cy, x2, y2)
        # pdb.set_trace()
        boxes[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        # (cx, cy, w, h)
        boxes[:, 2:] = (boxes[:, 2:] - boxes[:, :2]) * 2.0
        refined_anchors[ind] = boxes
        #nan_flags = (boxes != boxes)
        #if len(nan_flags.nonzero()) > 0:
        #    pdb.set_trace()
        #a = 0 
    return refined_anchors
