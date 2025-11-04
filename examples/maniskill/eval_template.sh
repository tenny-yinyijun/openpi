#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2                   # Number of tasks (processes)
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G                    # Memory per node
#SBATCH --time=72:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File


# -------------------------
# Parameters - These will be set by the submission script
# -------------------------

# Check if required parameters are provided
if [ -z "$ENV_ID" ] || [ -z "$POLICY_CONFIG" ] || [ -z "$POLICY_CHECKPOINT_DIR" ] || [ -z "$VIDEO_OUT_PATH" ] || [ -z "$NUM_EVALS" ]; then
    echo "Error: Missing required environment variables"
    echo "Required: ENV_ID, POLICY_CONFIG, POLICY_CHECKPOINT_DIR, VIDEO_OUT_PATH, NUM_EVALS"
    exit 1
fi

env_id="$ENV_ID"
policy_config="$POLICY_CONFIG"
policy_checkpoint_dir="$POLICY_CHECKPOINT_DIR"
video_out_path="$VIDEO_OUT_PATH"
num_evals="$NUM_EVALS"
save_data="${SAVE_DATA:-false}"  # Default to false if not set
save_eval_video="${SAVE_EVAL_VIDEO:-true}"  # Default to true if not set
seed="${SEED:-5000}"  # Default to 5000 if not set

# -------------------------

host=$(hostname)
echo "Running on $host"
echo "Environment: $env_id"
echo "Policy config: $policy_config"
echo "Policy checkpoint: $policy_checkpoint_dir"
echo "Video output path: $video_out_path"
echo "Number of evaluations: $num_evals"
echo "Save data: $save_data"
echo "Save eval video: $save_eval_video"
echo "Seed: $seed"

export PYTHONUNBUFFERED=1

# Create temporary directory for synchronization
sync_dir="/tmp/maniskill_sync_${SLURM_JOB_ID}"
mkdir -p $sync_dir

# Cleanup function
cleanup() {
    rm -rf $sync_dir
}
trap cleanup EXIT

# -------------------------
# Terminal 1: Environment simulation
# -------------------------
(
  source ~/.bashrc
  cd /n/fs/tom-project/papers/openpi || exit

  # Activate conda environment
  export PATH=$(echo $PATH | sed -e 's|/n/fs/tom-project/papers/openpi/.venv/bin:||')
  conda activate maniskill

  # Signal that environment is ready
  touch $sync_dir/env_ready
  echo "Environment process: Waiting for policy server to be ready..."

  # Wait for policy server to be ready
  while [ ! -f $sync_dir/policy_ready ]; do
    sleep 0.1
  done

  echo "Environment process: Both processes ready, starting evaluation..."

  # Run environment
  # Build the base command
  cmd="CUDA_VISIBLE_DEVICES=0 python examples/maniskill/main_eval.py"
  cmd="$cmd --args.host $host"
  cmd="$cmd --args.env_id $env_id"
  cmd="$cmd --args.num_evals $num_evals"
  cmd="$cmd --args.video_out_path $video_out_path"
  cmd="$cmd --args.seed $seed"

  # Add optional flags
  if [ "$save_data" = "true" ]; then
    cmd="$cmd --args.save_data"
  fi

  if [ "$save_eval_video" = "true" ]; then
    cmd="$cmd --args.save_eval_video"
  fi

  # Execute the command
  eval $cmd

  # Signal completion
  touch $sync_dir/env_done
  echo "Environment process: Completed"

) &> slurm_outputs/${SLURM_JOB_NAME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_env.out &

env_pid=$!

# -------------------------
# Terminal 2: Policy server
# -------------------------
(
  source ~/.bashrc
  cd /n/fs/tom-project/papers/openpi || exit

  # Signal that policy server is ready
  touch $sync_dir/policy_ready
  echo "Policy server: Waiting for environment to be ready..."

  # Wait for environment to be ready
  while [ ! -f $sync_dir/env_ready ]; do
    sleep 0.1
  done

  echo "Policy server: Both processes ready, starting server..."

  # Run policy server in background
  CUDA_VISIBLE_DEVICES=1 \
    uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=$policy_config \
      --policy.dir=$policy_checkpoint_dir &

  policy_pid=$!

  # Monitor environment process and exit when it's done
  while kill -0 $env_pid 2>/dev/null; do
    sleep 1
  done

  echo "Policy server: Environment process ended, shutting down..."
  kill $policy_pid 2>/dev/null
  wait $policy_pid 2>/dev/null

) &> slurm_outputs/${SLURM_JOB_NAME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_policy.out &

# Wait for environment process to complete
wait $env_pid

echo "Evaluation complete"
