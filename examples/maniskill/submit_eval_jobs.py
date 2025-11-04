#!/usr/bin/env python3
"""
Submit evaluation jobs for multiple checkpoints in parallel.

Usage:
    python submit_eval_jobs.py --checkpoints_file checkpoints.txt --output_path /path/to/videos --num_evals 10 --env_id StackCube-v1
"""

import argparse
from datetime import datetime
import json
import pathlib
import subprocess
import sys


def load_checkpoint_config(checkpoint_json_path):
    """Load checkpoint configuration from JSON file."""
    with open(checkpoint_json_path, 'r') as f:
        config = json.load(f)
    return config['policy_checkpoints']


def read_checkpoint_aliases(aliases_file):
    """Read checkpoint aliases from file (one per line)."""
    with open(aliases_file, 'r') as f:
        aliases = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return aliases


def submit_job(alias, checkpoint_info, env_id, video_out_path, num_evals, save_data, save_eval_video, seed, script_dir):
    """Submit a single sbatch job for a checkpoint."""

    # Create a unique job name based on alias
    job_name = f"eval_{alias}"

    # Create video output directory for this checkpoint with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"{alias}_n{num_evals}_{timestamp}"
    checkpoint_video_path = pathlib.Path(video_out_path) / dir_name

    # Set environment variables for the sbatch script
    env_vars = {
        'ENV_ID': env_id,
        'POLICY_CONFIG': checkpoint_info['config'],
        'POLICY_CHECKPOINT_DIR': checkpoint_info['ckpt'],
        'VIDEO_OUT_PATH': str(checkpoint_video_path),
        'NUM_EVALS': str(num_evals),
        'SAVE_DATA': 'true' if save_data else 'false',
        'SAVE_EVAL_VIDEO': 'true' if save_eval_video else 'false',
        'SEED': str(seed),
    }

    # Build the sbatch command
    cmd = [
        'sbatch',
        f'--job-name={job_name}',
        '--export=ALL,' + ','.join([f'{k}={v}' for k, v in env_vars.items()]),
    ]

    # Add the script path
    template_script = script_dir / 'eval_template.sh'
    cmd.append(str(template_script))

    # Submit the job
    print(f"Submitting job for checkpoint: {alias}")
    print(f"  Config: {checkpoint_info['config']}")
    print(f"  Checkpoint: {checkpoint_info['ckpt']}")
    print(f"  Video output: {checkpoint_video_path}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  Job submitted successfully: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error submitting job: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Submit evaluation jobs for multiple checkpoints'
    )
    parser.add_argument(
        '--checkpoints_file',
        type=str,
        required=True,
        help='Path to file containing checkpoint aliases (one per line)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Base path to save evaluation videos'
    )
    parser.add_argument(
        '--num_evals',
        type=int,
        required=True,
        help='Number of evaluations per checkpoint'
    )
    parser.add_argument(
        '--env_id',
        type=str,
        default='StackCube-v1',
        help='Environment ID (default: StackCube-v1)'
    )
    parser.add_argument(
        '--checkpoint_json',
        type=str,
        default=None,
        help='Path to checkpoint JSON file (default: examples/maniskill/pi_checkpoints.json)'
    )
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save raw data (states, actions, videos) in JSON and MP4 format'
    )
    parser.add_argument(
        '--save-eval-videos',
        action='store_true',
        help='Save evaluation videos (e.g., StackCube-v1_rollout_0_success.mp4). By default, videos are NOT saved.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=5000,
        help='Random seed for environment initialization (default: 5000). Episode i will use seed + i.'
    )

    args = parser.parse_args()

    # Determine script directory
    script_dir = pathlib.Path(__file__).parent

    # Set default checkpoint JSON path if not provided
    if args.checkpoint_json is None:
        args.checkpoint_json = script_dir / 'pi_checkpoints.json'

    # Load checkpoint configuration
    print(f"Loading checkpoint configuration from: {args.checkpoint_json}")
    try:
        checkpoint_configs = load_checkpoint_config(args.checkpoint_json)
    except FileNotFoundError:
        print(f"Error: Checkpoint JSON file not found: {args.checkpoint_json}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in checkpoint file: {e}")
        sys.exit(1)

    # Read checkpoint aliases
    print(f"Reading checkpoint aliases from: {args.checkpoints_file}")
    try:
        aliases = read_checkpoint_aliases(args.checkpoints_file)
    except FileNotFoundError:
        print(f"Error: Checkpoints file not found: {args.checkpoints_file}")
        sys.exit(1)

    if not aliases:
        print("Error: No checkpoint aliases found in the file")
        sys.exit(1)

    print(f"Found {len(aliases)} checkpoint(s) to evaluate")
    print()

    # Create output directory
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Submit jobs
    successful_submissions = 0
    failed_submissions = 0

    for alias in aliases:
        if alias not in checkpoint_configs:
            print(f"Warning: Checkpoint alias '{alias}' not found in configuration, skipping")
            failed_submissions += 1
            continue

        checkpoint_info = checkpoint_configs[alias]
        success = submit_job(
            alias=alias,
            checkpoint_info=checkpoint_info,
            env_id=args.env_id,
            video_out_path=args.output_path,
            num_evals=args.num_evals,
            save_data=getattr(args, 'save_data', False),
            save_eval_video=getattr(args, 'save_eval_videos', False),
            seed=args.seed,
            script_dir=script_dir
        )

        if success:
            successful_submissions += 1
        else:
            failed_submissions += 1

        print()

    # Summary
    print("=" * 80)
    print(f"Job submission complete:")
    print(f"  Successful: {successful_submissions}")
    print(f"  Failed: {failed_submissions}")
    print(f"  Total: {len(aliases)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
