# encoding: utf-8
import torch
import cv2
import numpy as np
import pdb

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (tensor) [batch, num_gt, 5]
            batch of annotations stacked on their 0 dim
            annotations for a given image are stacked on 1 dim
    """
    targets = []
    imgs = []
    # numpy array
    num_gts = [sample[1].shape[0] for sample in batch]
    max_num_gt = max(num_gts)
    for sample in batch:
        imgs.append(sample[0])
        size_gt = sample[1].shape
        num_gt = size_gt[0]
        aug_size = list(size_gt[:])
        aug_size[0] = max_num_gt
        aug_gt = np.zeros(aug_size, dtype=sample[1].dtype)
        aug_gt[:num_gt] = sample[1]
        targets.append(torch.FloatTensor(aug_gt))
    return torch.stack(imgs, 0), torch.stack(targets, 0)


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    """
    For evaluation and testing.
    """
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
