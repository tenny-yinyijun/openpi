# Parallel Evaluation Job Submission

Submit and run multiple ManiSkill policy evaluations in parallel using SLURM.

## Quick Start

```bash
# 1. Create a file listing checkpoint aliases (one per line, # for comments)
cat > my_checkpoints.txt <<EOF
stack_2000
stack_4000
EOF

# 2. Submit evaluation jobs
python examples/maniskill/submit_eval_jobs.py \
    --checkpoints_file my_checkpoints.txt \
    --output_path data/eval_results \
    --num_evals 10 \
    --env_id StackCube-v1
```

Each checkpoint will run as a separate SLURM job, evaluating N episodes in parallel.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoints_file` | File with checkpoint aliases (one per line) | Required |
| `--output_path` | Base directory for evaluation outputs | Required |
| `--num_evals` | Number of evaluation episodes per checkpoint | Required |
| `--env_id` | ManiSkill environment ID | `StackCube-v1` |
| `--checkpoint_json` | Path to checkpoint config JSON | `examples/maniskill/pi_checkpoints.json` |
| `--save-eval-video` | Save evaluation videos (e.g., `*_success.mp4`) | `True` |
| `--save-data` | Save raw data (states, actions, videos in JSON/MP4) | `False` |
| `--seed` | Random seed (episode i uses seed + i) | `5000` |

## Output Structure

Outputs are saved in timestamped directories with the format: `{alias}_n{num_evals}_{timestamp}`

```
<output_path>/
└── stack_2000_n10_20251102-143025/
    ├── StackCube-v1_rollout_0_success.mp4  # Evaluation videos (if --save-eval-video)
    ├── StackCube-v1_rollout_1_failure.mp4
    ├── evaluation_results.txt              # Success rate summary
    ├── videos_three_view_vertical/         # Raw camera views (if --save-data)
    │   └── 0.mp4
    └── json/                               # States/actions data (if --save-data)
        └── 0.json
```

### Evaluation Results

The `evaluation_results.txt` file contains:
- Overall success rate
- Per-episode results (success/failure and step count)

Example:
```
EVALUATION SUMMARY
================================================================================
Environment: StackCube-v1
Total episodes: 10
Successes: 8
Failures: 2
Success Rate: 80.00%
================================================================================

Per-episode results:
  Episode 1: SUCCESS (45 steps)
  Episode 2: FAILURE (200 steps)
  ...
```

## Key Features

- **Early stopping**: Episodes terminate on task success (rather than always running max timesteps)
- **Labeled videos**: Videos include `_success` or `_failure` suffix for easy sorting
- **Timestamped outputs**: Each run gets a unique directory with timestamp
- **Parallel execution**: Multiple checkpoints evaluated simultaneously via SLURM

## Configuration

### Adding Checkpoints

Edit `pi_checkpoints.json`:
```json
{
    "policy_checkpoints": {
        "my_checkpoint": {
            "config": "policy_config_name",
            "ckpt": "/path/to/checkpoint/dir"
        }
    }
}
```

Then reference `my_checkpoint` in your checkpoints file.

## Examples

### Basic Evaluation
```bash
python examples/maniskill/submit_eval_jobs.py \
    --checkpoints_file my_checkpoints.txt \
    --output_path data/eval \
    --num_evals 20 \
    --env_id StackCube-v1
```

### Save Raw Data for Analysis
```bash
python examples/maniskill/submit_eval_jobs.py \
    --checkpoints_file my_checkpoints.txt \
    --output_path data/eval \
    --num_evals 10 \
    --save-data
```

### Custom Seed
```bash
python examples/maniskill/submit_eval_jobs.py \
    --checkpoints_file my_checkpoints.txt \
    --output_path data/eval \
    --num_evals 10 \
    --seed 1000  # Episodes will use seeds 1000-1009
```

### Skip Saving Videos
```bash
python examples/maniskill/submit_eval_jobs.py \
    --checkpoints_file my_checkpoints.txt \
    --output_path data/eval \
    --num_evals 10 \
    --no-save-eval-video  # Don't save evaluation videos
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# View job output
tail -f slurm_outputs/eval_stack_2000/out_log_eval_stack_2000_<job_id>.out
```

## Files

- `submit_eval_jobs.py`: Job submission script
- `eval_template.sh`: SLURM batch template
- `main_eval.py`: Evaluation script with success tracking
- `pi_checkpoints.json`: Checkpoint configuration
- `example_checkpoints.txt`: Example checkpoint list
