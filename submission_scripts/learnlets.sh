#!/bin/bash
module load GCC/8.3.0 CUDA/10.1.243 cuDNN/7.6.4.38 NCCL/2.7.3
conda activate learnlet

export BSD500_DATA_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
export BSD68_DATA_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
export DIV2K_DATA_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
export LOGS_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
export CHECKPOINTS_DIR='/home/users/a/akhaury/Thesis/understanding-unets/'
