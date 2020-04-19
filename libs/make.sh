#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

export CUDA_PATH=/usr/local/cuda/
#You may also want to ad the following
#export C_INCLUDE_PATH=/opt/cuda/include

export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

python setup.py build_ext --inplace
rm -rf build
