# Maniskill Benchmark

## Running Pi in Maniskill

```bash
# In terminal 1, run:
conda activate maniskill

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

## Finetuning Pi on Maniskill Dataset

First, run dataset conversion

```bash
uv run examples/maniskill/convert_maniskill_data_to_lerobot.py
```

Next, define the config name (e.g. `pi0_maniskill_t0_test_low_mem_finetune`) and compute norm states:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_maniskill_t0_test_low_mem_finetune
```

Finally, start training:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_maniskill_t0_test_low_mem_finetune --exp-name=my_experiment --overwrite
```