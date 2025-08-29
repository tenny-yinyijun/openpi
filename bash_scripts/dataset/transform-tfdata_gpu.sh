#!/bin/bash
#SBATCH --job-name=my_gpu_job        # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=80G                    # Memory per node
#SBATCH --time=10:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --exclude=neu301

echo "running compute_norm_stats on pi0_fast_libero_low_mem_finetune"

export PYTHONUNBUFFERED=1

cd /n/fs/tom-project/papers/openpi

uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune