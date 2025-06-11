#!/bin/bash

#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --partition=<your_partitions>
#SBATCH --account=<your_account>
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --job-name=test_segmentation_models

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

NAME="test_segmentation_models"
IMAGE="<your_image>"  # provide your image here

## tiny_model
CONFIG=./configs/mamba_vision/mamba_vision_160k_ade20k-512x512_tiny.py
CHECKPOINT=./ckpts/mamba_vision_160k_ade20k-512x512_tiny.pth # provide the checkpoint

## base_model
CONFIG=./configs/mamba_vision/mamba_vision_160k_ade20k-512x512_base.py
CHECKPOINT=./ckpts/mamba_vision_160k_ade20k-512x512_base.pth # provide the checkpoint

## large model
CONFIG=./configs/mamba_vision/mamba_vision_160k_ade20k-640x640_large_21k.py
CHECKPOINT=./ckpts/mamba_vision_160k_ade20k-640x640_large_21k.pth # provide the checkpoint

## l3 model
CONFIG=./configs/mamba_vision/mamba_vision_160k_ade20k-640x640_l3_21k.py
CHECKPOINT=./ckpts/mamba_vision_160k_ade20k-640x640_l3_21k.pth # provide the checkpoint

OUTPUT_ROOT="./tools"
export PYTHONPATH="${OUTPUT_ROOT}":$PYTHONPATH

SAVE_DIR="./logs"
LOGS_DIR="${SAVE_DIR}/logs/${NAME}/"

run_cmd="python -u ${OUTPUT_ROOT}/test.py ${CONFIG} ${CHECKPOINT} --launcher "slurm""

mkdir -p ${LOGS_DIR}

DATETIME=`date +'date_%Y-%m-%d_time_%H-%M-%S'`

srun -l \
     --container-image=${IMAGE} \
     --container-mounts=/lustre:/lustre,/home/${USER}:/home/${USER} \
     --container-workdir=${OUTPUT_ROOT} \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${run_cmd}"

set +x