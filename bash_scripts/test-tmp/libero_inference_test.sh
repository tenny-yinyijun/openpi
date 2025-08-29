# first srun: start eval script
# change host in main.py to name of node in second srun
cd /n/fs/tom-project/papers/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main.py

# second srun: start policy
cd /n/fs/tom-project/papers/openpi
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero_low_mem_finetune --policy.dir=checkpoints/pi0_fast_libero_low_mem_finetune/20250827-155704-4gpu/6000


# collect failure data
cd /n/fs/tom-project/papers/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/collect_rollout.py --args.host neu322 --args.expname "0828-test0-2000"

cd /n/fs/tom-project/papers/openpi
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero_low_mem_finetune --policy.dir=checkpoints/pi0_fast_libero_low_mem_finetune/20250828-060308-4gpu/4000