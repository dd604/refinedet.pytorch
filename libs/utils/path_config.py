from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

cfg = edict()

# Root directory of project
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                                    '..', '..'))
jarvis = False 
# Data directory
if jarvis:
    cfg.DATA_DIR = os.getenv('DATA_DIR')
    output_root = os.getenv('OUTPUT_DIR')
    cfg.OUTPUT_DIR = os.path.join(output_root, 'refinedet/output')
    cfg.CACHE_ROOT = os.path.join(output_root, 'refinedet/cache')
    # model_dir = os.getenv('MODEL_DIR')
    # print('model_dir', model_dir)
else:
    cfg.DATA_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, 'data'))
    cfg.CACHE_ROOT = cfg.DATA_DIR
    cfg.OUTPUT_DIR = cfg.ROOT_DIR

# print('Must create data folder in the root of the project.')
# Name (or path to) the matlab executable
cfg.MATLAB = 'matlab'

print(cfg)
