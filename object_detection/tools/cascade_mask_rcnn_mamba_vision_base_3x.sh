#!/bin/bash

#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --partition=<your_partitions>
#SBATCH --account=<your_account>
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --job-name=cascade_mambavision_base

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

NAME="cascade_mambavision_base"
IMAGE="<your_image>"  # provide your image here

CONFIG=./configs/mamba_vision/cascade_mask_rcnn_mamba_vision_base_3x_coco.py

OUTPUT_ROOT="./tools"
export PYTHONPATH="${OUTPUT_ROOT}":$PYTHONPATH

SAVE_DIR="./logs"
LOGS_DIR="${SAVE_DIR}/logs/${NAME}/"

run_cmd="python -u ${OUTPUT_ROOT}/train.py ${CONFIG} --launcher "slurm""
#If you want to continue training or resume from a checkpoint, just add --resume to run_cmd

mkdir -p ${LOGS_DIR}

DATETIME=`date +'date_%Y-%m-%d_time_%H-%M-%S'`

srun -l \
     --container-image=${IMAGE} \
     --container-mounts=/lustre:/lustre,/home/${USER}:/home/${USER} \
     --container-workdir=${OUTPUT_ROOT} \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${run_cmd}"

set +x

