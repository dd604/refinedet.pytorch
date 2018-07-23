# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [40, 20, 10, 5],
    'min_dim': 320,
    'steps': [8, 16, 32, 64],
    'min_sizes': [30, 60, 111, 162],
    'max_sizes': [60, 111, 162, 213],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

#
coco = {
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000), # ok
    'max_iter': 400000, # ok
    'feature_maps': [40, 20, 10, 5], # ok
    'min_dim': 320, # ok
    'steps': [8, 16, 32, 64], # ok
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'mbox': [3, 3, 3, 3],  # number of boxes per feature map location
    # 'variance': [0.1, 0.1, 0.2, 0.2],
    'variance': [0.1, 0.2],
    'clip': False,
    'name': 'COCO',
}

