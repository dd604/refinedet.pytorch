from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

cfg = edict()

# Root directory of project
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                                    '..', '..'))

# Data directory
cfg.DATA_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, 'data'))
# print('Must create data folder in the root of the project.')
# Name (or path to) the matlab executable
cfg.MATLAB = 'matlab'