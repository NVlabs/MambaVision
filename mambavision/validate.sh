#!/bin/bash
DATA_PATH="/ImageNet/val"
BS=128
checkpoint='/model_weights/mambavision_tiny_1k.pth.tar'

python validate.py --model mamba_vision_T --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch-size $BS --input-size 3 224 224

