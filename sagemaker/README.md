# openpi TRI/LBM bimanual fine-tuning on SageMaker (cv-wfm / H200)

Full fine-tune of **pi0_base** / **pi05_base** on TRI/LBM bimanual dual-Panda tasks. Configs
live in `src/openpi/training/config.py`:

| task (raw dir)                 | dataset repo_id                  | config(s)                            | teleop eps | frames  |
| ------------------------------ | -------------------------------- | ------------------------------------ | ---------- | ------- |
| `BimanualBikeRotorInstall`     | `tri/bike_rotor_cartesian`       | `pi0_bike_rotor`, `pi05_bike_rotor`  | 534        | 519.8k  |
| `BimanualSetUpBreakfastTable`  | `tri/breakfast_table_cartesian`  | `pi05_breakfast_table`               | 341        | 278.7k  |
| `BimanualCleanUpSpill`         | `tri/clean_spill_cartesian`      | `pi05_clean_spill`                   | 151        | 54.2k   |

All three come off the same rig and are byte-compatible in shape, so they share
`LeRobotBikeRotorDataConfig` and `bike_rotor_policy`'s transforms; only the dataset (and hence
the norm stats) differ. Bike rotor and clean spill were recorded at **ruggles**, breakfast
table at **hersey** — the station changes nothing downstream.

- **Cameras** (pi0's 3 slots): `base_0_rgb`=scene_right_0, `left_wrist_0_rgb`=wrist_left_plus,
  `right_wrist_0_rgb`=wrist_right_plus.
- **State**: 16-d measured joint state (jpos L/R + gripper L/R).
- **Actions**: 20-d cartesian `xyzrot6g` (absolute EE pose+gripper per arm). No delta transform;
  normalization handles scale.
- **Data**: `teleop` demos from the base task dir only. `rollout` episodes (policy eval, many
  unsuccessful) are excluded, and so are the `<task>_v?_DAgger?_DAggerType_*` sibling
  directories — those hold human teleop under deliberately perturbed initial conditions
  (807 more teleop episodes for clean spill, 453 for bike rotor, none for breakfast table).
  Excluding them is consistent with how `pi05_bike_rotor` was trained;
  add them by pointing `--task` at those dirs and merging if you want DAgger coverage.

## One-time local prep (per task; do this before launching)

```bash
cd ~/workspace/openpi

# 1. Convert raw LBM teleop demos -> LeRobot dataset (h264 video).
#    Parallel: 32 shard processes over disjoint episodes, then auto-merge into one dataset.
#    (Single-process fallback: convert_bike_rotor_to_lerobot.py --repo-id ... [--task ...])
bash examples/bike_rotor/run_conversion_parallel.sh tri/bike_rotor_cartesian 32
bash examples/bike_rotor/run_conversion_parallel.sh tri/breakfast_table_cartesian 32 \
    --task BimanualSetUpBreakfastTable
bash examples/bike_rotor/run_conversion_parallel.sh tri/clean_spill_cartesian 32 \
    --task BimanualCleanUpSpill

# 2. Compute normalization stats straight from the raw lowdim (no video decode -> no local
#    ffmpeg/torchcodec dependency, and much faster). Writes
#    ./assets/<config>/<repo_id>/norm_stats.json, baked into the image. Stats are per task:
#    --task, --repo-id and --configs must move together.
#    (Equivalent to scripts/compute_norm_stats.py for these configs: we apply no delta transform.)
uv run examples/bike_rotor/compute_norm_stats_from_raw.py
uv run examples/bike_rotor/compute_norm_stats_from_raw.py \
    --task BimanualSetUpBreakfastTable --repo-id tri/breakfast_table_cartesian \
    --configs pi05_breakfast_table
uv run examples/bike_rotor/compute_norm_stats_from_raw.py \
    --task BimanualCleanUpSpill --repo-id tri/clean_spill_cartesian \
    --configs pi05_clean_spill

# If you prefer the stock path instead, it needs a working video backend (system ffmpeg):
#   uv run scripts/compute_norm_stats.py --config-name pi0_bike_rotor

# 3. Stage dataset + base checkpoints to S3 (mounted offline by the jobs).
bash sagemaker/stage_to_s3.sh tri/bike_rotor_cartesian
```

## Launch (submits to the cv-wfm p5en H200 queue)

```bash
export SM_USER=tenny            # namespaces the ECR repo / job / S3 paths
export WANDB_API_KEY=...        # or have `wandb login` populate ~/.netrc

# The first launch builds+pushes the image; reuse it afterwards with BUILD_TYPE=None:
bash sagemaker/run_sm.sh pi0_bike_rotor       bike-rotor-pi0   cv-wfm full
bash sagemaker/run_sm.sh pi05_bike_rotor      bike-rotor-pi05  cv-wfm None
bash sagemaker/run_sm.sh pi05_breakfast_table breakfast-pi05   cv-wfm None
bash sagemaker/run_sm.sh pi05_clean_spill     clean-spill-pi05 cv-wfm None
```

Add `--dry-run` (as an extra flag) to print the payload without submitting. Monitor with
open-world's `sagemaker/sqm.sh` (these are AWS Batch service-jobs).

## Running locally instead

These fit on 80 GB-class GPUs without SageMaker (base params are cached in
`~/.cache/openpi/openpi-assets/checkpoints`, so no GCS egress either) — useful when the queue
is busy. A full pi05 fine-tune at `batch_size=32` fits on as few as **2** GPUs via FSDP
(measured on the DGX: 4 GPUs → 1.0 s/it, 2 GPUs → 1.8 s/it, i.e. ~8.5 h / ~15 h for 30k steps):

```bash
HF_DATASETS_CACHE=/tmp/tenny_hf_datasets \      # see below
OPENPI_VIDEO_BACKEND=pyav \                     # see below
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=1,2,4,5 \
  uv run scripts/train.py pi05_breakfast_table --exp-name breakfast_table_pi05_v1 --fsdp-devices 4
```

Checkpoints land in `./checkpoints/<config>/<exp-name>/`. Two host quirks the docker image
hides, both of which abort the run before step 0:

- `--fsdp-devices` must divide `batch_size` (32), so use 1/2/4/8 GPUs, not 6.
- **`HF_DATASETS_CACHE`** off the NFS home: `datasets` refuses to build the parquet cache with
  "Not enough disk space" because `statvfs` on `/home` reports 0 free (a filer artefact — the
  filesystem writes fine). Any real local path fixes it; the cache is ~100 MB.
- **`OPENPI_VIDEO_BACKEND=pyav`**: lerobot defaults to torchcodec whenever it imports, and
  torchcodec then fails to load `libavutil.so.56/57/58` if the host has no ffmpeg 4-6 runtime.
  PyAV ships its own ffmpeg, so it decodes without system libs.

## Validation loss
Every config sets `val_fraction=0.05`: a deterministic **episode-level** 5% holdout (e.g. 27 of
534 bike episodes, seeded so train/val never share an episode → no frame leakage). Every
`val_interval` (1000) steps the trainer averages `num_val_batches` (20) no-grad batches and
logs `val_loss` to wandb (uses EMA params when available). Norm stats are computed over all
episodes (standard; the 5% holdout is negligible for stats). To disable, set `val_fraction=0`.

## Notes
- **Full fine-tune**: no LoRA variant, no freeze filter. pi0 uses z-score norm; pi05 uses
  quantile norm (set automatically by model type).
- Base weights are passed via `--weight-loader.params-path` (mounted `base_ckpt` channel),
  overriding the config's `gs://` path so training needs no GCS egress.
- Checkpoints land in `.../sagemaker/<user>/<job>/checkpoints`. `action_horizon=16`,
  `batch_size=32`, `num_train_steps=30000` — tune in the config as needed. At that fixed
  schedule the epoch count varies a lot by dataset (bike 1.8, breakfast 3.4, clean spill 17.7);
  clean spill is the one to watch for overfitting on `val_loss`.
- If a job OOMs, `--fsdp-devices` already defaults to the GPU count; also confirm
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.
