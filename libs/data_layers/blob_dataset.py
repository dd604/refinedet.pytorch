# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import cv2

import pdb


def normalize_boxes(boxes, width, height):
    scale = np.array([width, height, width, height],
                     dtype=np.float32)
    res = boxes / scale
    
    return res


class BlobDataset(data.Dataset):
    def __init__(self, imdb, roidb, transform=None,
                 target_normalization=True):
        """
        :param roidb:
        :param transform:
        :param target_normalization:
        :param class_names: with background, class from 0 - classnum.
        """
        self.imdb = imdb
        self.roidb = roidb
        self.transform = transform
        self.target_normalization = target_normalization
        self._class_names = imdb._classes
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        img, gt, h, w = self.pull_item(index)
        return img, gt
    
    def __len__(self):
        return len(self.roidb)
    
    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
        """
        blobs = self.roidb[index]
        # img_id = blobs['img_id']
        img_path = blobs['image']
        
        # pdb.set_trace()
        width = blobs['width']
        height = blobs['height']
        
        boxes = blobs['boxes']
        # idx from 1
        labels = blobs['gt_classes']
        # opencv image, since transfrom is used.
        # BGR
        img = cv2.imread(img_path)
        if self.target_normalization is not None:
            boxes = normalize_boxes(boxes, width, height)
        # target = np.vstack((boxes, labels))
        if self.transform is not None:
            img, boxes, labels = self.transform(img,
                                boxes, labels)
            target = np.hstack((boxes, np.expand_dims(
                labels, axis=1)))
        # [C, H, W]
        return torch.from_numpy(img).permute(2, 0, 1), target, \
                    height, width
        
        
    # def pull_image(self, index):
    #     '''Returns the original image object at index in PIL form
    #
    #     Note: not using self.__getitem__(), as any transformations passed in
    #     could mess up this functionality.
    #
    #     Argument:
    #         index (int): index of img to show
    #     Return:
    #         gbr image
    #     '''
    #     path = self.roidb[index]['image']
    #     img = np.array(Image.open(path))[:, :, (2, 1, 0)]
    #
    #     return img
    
    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
    
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_path, ([bbox coords, class_id], label_name),...]]
                eg: ('001718.jpg', ([96, 13, 438, 332, 15], 'dog')])
        '''
        
        blobs = self.roidb[index]
        result = [blobs['image']]
        labels = blobs['gt_classes']
        gt_array = np.hstack((blobs['boxes'], np.expand_dims(
            blobs['gt_classes'], dim=1)))
        gts = []
        label_names = []
        for ind in len(gt_array):
            gts.append(gt_array[ind, :])
            label_names.append(self._class_names[gt_array[ind, -1]])
        pairs = list(zip(gts, labe_names))
        result.append(pairs)
        
        return result
        
    
