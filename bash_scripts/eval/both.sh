#!/bin/bash
#SBATCH --job-name=evalb
#SBATCH --nodes=1 
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2                   # Number of tasks (processes)
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G                    # Memory per node
#SBATCH --time=72:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --exclude=neu301


# -------------------------
# Parameters
# -------------------------

policy_ckpt=2000


# -------------------------

host=$(hostname)
echo "Running on $host"
echo "Policy checkpoint: $policy_ckpt"

export PYTHONUNBUFFERED=1

# -------------------------
# Terminal 1: Simulation
# -------------------------
(
  cd /n/fs/tom-project/papers/openpi || exit
  source examples/libero/.venv/bin/activate
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
  CUDA_VISIBLE_DEVICES=0 \
    python examples/libero/collect_rollout.py \
      --args.host $host \
      --args.expname "0828-rollout-${policy_ckpt}" \
) &> slurm_outputs/${SLURM_JOB_NAME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_sim.out &

# -------------------------
# Terminal 2: Policy server
# -------------------------
(
  cd /n/fs/tom-project/papers/openpi || exit
  CUDA_VISIBLE_DEVICES=1 \
    uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=pi0_fast_libero_low_mem_finetune \
      --policy.dir=checkpoints/pi0_fast_libero_low_mem_finetune/20250828-060308-4gpu/${policy_ckpt}
) &> slurm_outputs/${SLURM_JOB_NAME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_policy.out &

wait