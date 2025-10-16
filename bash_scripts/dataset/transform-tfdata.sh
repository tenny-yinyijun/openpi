#!/bin/bash
#SBATCH --job-name=my_cpu_job        # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=160G                    # Memory per node
#SBATCH --time=10:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File

# Your commands here
echo "Running on CPU with $SLURM_CPUS_PER_TASK cores"

cd /n/fs/tom-project/papers/openpi

uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /n/fs/iromdata/modified_libero_rlds