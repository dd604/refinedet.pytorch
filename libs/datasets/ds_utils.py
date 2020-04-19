# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


def bbox_overlaps(bboxes, ref_bboxes):
    """
    ref_bboxes: N x 4;
    bboxes: K x 4

    return: K x N
    """
    refx1, refy1, refx2, refy2 = np.vsplit(np.transpose(ref_bboxes), 4)
    x1, y1, x2, y2 = np.hsplit(bboxes, 4)
    
    minx = np.maximum(refx1, x1)
    miny = np.maximum(refy1, y1)
    maxx = np.minimum(refx2, x2)
    maxy = np.minimum(refy2, y2)
    
    inter_area = (maxx - minx + 1) * (maxy - miny + 1)
    ref_area = (refx2 - refx1 + 1) * (refy2 - refy1 + 1)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    iou = inter_area / (ref_area + area - inter_area)
    
    return iou