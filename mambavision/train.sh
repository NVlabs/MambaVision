#!/bin/bash

DATA_PATH="/media/ali/ali/imagenet-a/"
MODEL=faster_vit_0_224
BS=2
EXP=Test
LR=8e-4
WD=0.05
WR_LR=1e-6
DR=0.38
MESA=0.25

torchrun --nproc_per_node=1 train.py --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
--data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR
