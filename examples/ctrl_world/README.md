# Interaction with Ctrl World

## Install as package

```bash
# Activate your world model environment, then run:

cd path/to/openpi
pip install -e packages/openpi-client

cd path/to/Ctrl-World
python -m pip install -e .
```

## Running Interaction Loop

```bash
# Terminal 1: Policy (20GB)
uv run scripts/serve_policy.py --env=DROID05
```

```bash
# Terminal 2: WM
conda activate ctrl-world
python examples/ctrl_world/main.py --args.host <node_name>
```