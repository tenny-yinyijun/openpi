#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:1                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=80G                                       ## Memory per node
#SBATCH --time=72:00:00                                 ## Walltime
#SBATCH --job-name=pi0lib                                 ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=yy4041@princeton.edu
#SBATCH --exclude=neu301

source ~/.bashrc

cd /n/fs/tom-project/papers/openpi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export RUN_DATETIME=$(date +%Y%m%d-%H%M%S)
export EXPERIMENT_NAME="$RUN_DATETIME-1gpu"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_low_mem_finetune --exp-name=$EXPERIMENT_NAME --overwrite