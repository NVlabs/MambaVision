#!/bin/bash
DATA_PATH="/home/ali/Desktop/data_local/ImageNet-Validation/val"
BS=128
checkpoint='/home/ali/Desktop/Vision_Transformers/model_weights/faster_vit_models/fastervit_0_224_1k/fastervit_0_224_1k.pth.tar'

python validate.py --model faster_vit_0_any_res --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch-size $BS --input-size 3 224 224

