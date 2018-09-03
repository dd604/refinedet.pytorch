import os
import sys

root_dir = os.path.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = os.path.join(root_dir, 'lib')
sys.path.append(lib_path)

# coco_path = os.path.join(root_dir, 'data', 'coco', 'PythonAPI')
# add_path(coco_path)
