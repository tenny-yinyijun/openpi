#!/bin/bash
#SBATCH --job-name=evalp        # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=80G                    # Memory per node
#SBATCH --time=72:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --exclude=neu301

source ~/.bashrc

cd /n/fs/tom-project/papers/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

python examples/libero/eval/collect_data.py \
    --args.host neu322  \
    --args.num_trials_per_task 50 \
    --args.expname "0828-test0-2000"
