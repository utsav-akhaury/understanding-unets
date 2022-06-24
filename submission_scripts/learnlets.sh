#!/bin/bash
module load GCCcore/10.3.0 CUDA/11.3.1 cuDNN/8.2.1.32-CUDA-11.3.1 NCCL/2.10.3-CUDA-11.3.1
conda activate learnlet

export BSD500_DATA_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
export BSD68_DATA_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
export DIV2K_DATA_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
export LOGS_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
export CHECKPOINTS_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
