#!/bin/bash
DATA_PATH="/ImageNet/val"
BS=128

python validate_pip_model.py --model mamba_vision_T --data_dir=$DATA_PATH --batch-size $BS

