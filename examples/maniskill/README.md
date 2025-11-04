# Maniskill Benchmark

## Running Pi in The World Model

Setup (ran only once): Install the following in your world model environment:
```bash
# Install openpi client
cd path/to/openpi
pip install -e packages/openpi-client

# Install maniskill
cd path/to/maniskill
python -m pip install -e .
```

We provide one policy checkpoint fine-tuned on the maniskill environment, and one world model checkpoint also fine-tuned on the same environment. To interface the two:

```bash
# In terminal 1
conda activate wmenv
python examples/maniskill/main_wm.py --args.host <node_name>

# In terminal 2

# pick-place-droid
uv run scripts/serve_policy.py policy:checkpoint --policy.config pi0_maniskill_all_jointpos_lora --policy.dir /n/fs/tom-project/papers/openpi/checkpoints/pi0_maniskill_all_jointpos_lora/20251021-205853-4gpu/50000

# stack
uv run scripts/serve_policy.py policy:checkpoint --policy.config pi0_maniskill_stack400_pd_joint_pos_lora --policy.dir /n/fs/tom-project/papers/openpi/checkpoints/pi0_maniskill_stack400_pd_joint_pos_lora/20251102-013718-4gpu/14000

```

## Running Pi in Maniskill (Single run)

Setup (ran only once): Install openpi-client in maniskill using `pip install -e packages/openpi-client`.

```bash
conda activate maniskill

# only run once:
cd path/to/openpi
pip install -e packages/openpi-client

# start evaluation
python examples/maniskill/main.py --args.host <node_name> --args.env_id <env_id_name>
```

If `mani_skill is not found`:

```bash
# Make sure this points to the correct python path
which python

# if the path is under openpi instead, remove it with something like:
export PATH=$(echo $PATH | sed -e 's|/path/to/openpi/.venv/bin:||')

```

```bash
# In terminal 2, run:
## Running your own checkpoint
uv run scripts/serve_policy.py policy:checkpoint --policy.config <policy_config_name> --policy.dir <policy_checkpoint_dir> 

## Running the droid policy
uv run scripts/serve_policy.py --env DROID
```

## Running Pi in Maniskill (Batch experiment)

```bash
python examples/maniskill/submit_eval_jobs.py \
    --checkpoints_file examples/maniskill/my_checkpoints.txt \
    --output_path /n/fs/tom-project/papers/openpi/examples/maniskill/test_outputs/pi_checkpoints_4000_v2 \
    --num_evals 50 \
    --env_id StackCube-v1 \
    --seed 1000

# for data generation
python examples/maniskill/submit_eval_jobs.py \
    --checkpoints_file examples/maniskill/rollout_checkpoints.txt \
    --output_path /n/fs/iromdata/video_model_training/maniskill/rollouts \
    --num_evals 200 \
    --env_id StackCube-v1 \
    --seed 1000 \
    --save-data 
```

## Finetuning Pi on Maniskill Dataset

First, run dataset conversion ([sbatch version](bash_scripts/dataset/maniskill_convert.sh))

```bash
uv run examples/maniskill/convert_maniskill_data_to_lerobot.py
```

Next, define the config name (e.g. `pi0_maniskill_t0_test_low_mem_finetune`) and compute norm states: ([sbatch version](bash_scripts/dataset/maniskill_normstats.sh))

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_maniskill_t0_test_low_mem_finetune
```

Finally, start training: ([sbatch version](bash_scripts/dataset/maniskill_normstats.sh))

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_maniskill_t0_test_low_mem_finetune --exp-name=my_experiment --overwrite
```