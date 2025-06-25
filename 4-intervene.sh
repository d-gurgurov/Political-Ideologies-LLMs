#!/bin/bash

pip install git+https://github.com/davidbau/baukit
pip install -U torch accelerate transformers numba
pip install numpy==1.26.4

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python intervene.py