# pi0.5 → anzu gRPC bridge

Lets `~/multiarm/anzu`'s existing `rollout_grpc` client (davis's
`DiffusionHwToGrpc` scenario) drive a **pi0.5 / openpi** checkpoint instead of
VLA Foundry, with zero changes on the anzu/station side. Built for
[`tennyyyin/pi05-clean-spill-dagger`](https://huggingface.co/tennyyyin/pi05-clean-spill-dagger).

Lives in this fork (alongside `examples/bike_rotor`) rather than inside
`~/lbm` or `~/multiarm/anzu`: those are the lab's shared checkouts, and this
only ever *imports* from them via `PYTHONPATH`, never edits them. Keeping it
here also keeps the hardware-eval path in the same repo as the checkpoints'
training config (`pi05_clean_spill`).

## Why a bridge is needed at all

VLA Foundry and pi0.5/openpi are different serving stacks:

| | VLA Foundry | pi0.5/openpi |
|---|---|---|
| Server protocol | gRPC on `:50051`, anzu's native protocol | openpi's own WebSocket server (`serve_policy.py`) |
| Cameras | 6 (davis's full set) | 3: `observation/image`, `observation/left_wrist_image`, `observation/right_wrist_image` |
| Proprioception | — | joint positions, not poses |
| Action | `PosesAndGrippers` | `[16, 20]` absolute-pose chunk |

`pi0_policy_server.py` + `pi0_policy_wrapper.py` implement the missing piece:
a `robot_gym.policy.Policy` (same interface `diffusion_policy_server.py` /
`vla_policy_server.py` implement) that is actually a WebSocket client to your
running `serve_policy.py`, re-served over gRPC for anzu.

## Status

- **Code**: run end-to-end on `davis` on 2026-09-01 — anzu's `rollout_grpc`
  client driving both Pandas off a pi0.5 checkpoint through this bridge.
  Follows the existing `DiffusionPolicyConfig` / `VLAPolicyWrapperConfig`
  pattern in `lbm/grpc_workspace`.
- **Checkpoints**: served from a local directory (`--policy.dir`); the one
  exercised so far is `~/tennyyin/ckpts/base`. `--policy-config` stays
  `pi05_clean_spill` for all four clean-spill checkpoints.
- **`davis.txt`**: `BimanualCleanUpSpill` must NOT be added to
  `~/multiarm/anzu/intuitive/skill_types/davis.txt`. The skill enum is global
  across every station's `*.txt`, so its existing `ruggles.txt` entry already
  makes it valid with `--station davis`; adding it a second time raises
  `ValueError: Duplicates in skill type lists`.
- **Camera mapping**: CONFIRMED by tennyyyin on 2026-09-01, against the
  running station — `base_camera=scene_right_0`,
  `left_wrist_camera=wrist_left_plus`, `right_wrist_camera=wrist_right_plus`.
  These are now the defaults in `Pi0PolicyWrapperConfig` /
  `pi0_policy_server.py`, no flags needed for the standard case.
- **State vector ordering** (14 joints + 2 grippers): CONFIRMED **left-first**
  (`left_joint_0..6, right_joint_0..6, left_gripper, right_gripper`) against
  `examples/bike_rotor/convert_bike_rotor_to_lerobot.py`, the training-data
  generation source. The 20-d *action* is right-first — the two do not share
  an arm-order convention. An earlier version of this bridge assumed
  right-first for both and produced garbled proprioception with no error;
  `test_pi0_bridge_smoke.py` now pins the ordering.
- **Smoothing**: absolute-pose chunks need multi-draw averaging plus temporal
  ensembling to be usable; see `--samples-per-call`, `--ensemble-decay`,
  `--prefetch-lead-ticks` and `analyze_trajectory.py`'s seam report, which is
  the metric to re-run after touching any of them.

## Run

```bash
# 1) openpi's own server, from your fork checkout
cd <openpi-fork-checkout>
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_clean_spill --policy.dir=<ckpt-dir>

# 2) this bridge, from openpi's OWN venv (it already has openpi_client).
#    One-time: install anzu's exact policy_interfaces wheel, or the gRPC
#    package names skew and anzu gets `UNIMPLEMENTED: Method not found!`
#    -- see pi0_policy_server.py's docstring.
cd <openpi-fork-checkout>
uv pip install ~/multiarm/anzu/tools/workspace/venv/wheels/policy_interfaces-0.1.4-py3-none-any.whl
# --no-sync is REQUIRED: a plain `uv run` re-syncs the venv and undoes that
# install. PYTHONPATH only needs lbm (for prismatic/diffusion_policy/
# robot_gym); this directory lands on sys.path[0] automatically.
PYTHONPATH=~/lbm uv run --no-sync python examples/pi0_grpc_bridge/pi0_policy_server.py \
    --checkpoint-path <ckpt-dir-or-s3-path> \
    --skill-type BimanualCleanUpSpill
# camera flags default to the confirmed mapping (scene_right_0 /
# wrist_left_plus / wrist_right_plus); pass --base-camera etc. to override.
# wait for: Started Server loop on localhost:50051...

# 3) anzu rollout, unchanged from any other gRPC-served policy
cd ~/multiarm/anzu
AWS_PROFILE=sagemaker ./run --build "" \
    //intuitive/visuomotor/demo:quick_run_visuomotor_experiment \
    --skill BimanualCleanUpSpill --station davis \
    --demonstration_indices 0:100 --mode=rollout_grpc \
    --policy_type=diffusion --save_async --operator_name "test" \
    --language_instruction "pick up the knocked over cup, set it upright, and wipe up the spilled liquid with a towel"
```

Step 3 needs `~/efs/data/tasks/<skill>/eval_info/<skill>_rubric.txt` to exist,
or anzu dies at the end of the first episode when the success/failure GUI is
filled in (`AttributeError: 'NoneType' object has no attribute
'set_ready_to_be_read'` -- `preload_rubric_gui` only warns on a missing
template, but `fill_rubric_via_gui` is called regardless). Copy
`BimanualTemplate_rubric.txt` and add task-specific Y/N lines.

## What a run produces

Each episode is saved to
`~/tennyyin/data/<checkpoint-basename>_<task-shorthand>/<timestamp>/`: one
`.mp4` per model-input camera view, plus `states.csv`/`states.jsonl`
(commanded and reached, one row per control tick) and `meta.json`. See
`episode_recorder.py`; `--no-record` turns it off. Encoding is H.264 via
**PyAV** (in-process libavcodec, pulled in by openpi's `lerobot` dependency)
rather than `imageio`/the ffmpeg CLI — nothing in this process may fork while
anzu holds a gRPC connection to it, or the connection dies mid-rollout. See
rule 3 in `episode_recorder.py`'s docstring; the self-test asserts it.
Separately, `--log-path`
appends every tick across all episodes to `trajectory_logs/` (gitignored) for
`analyze_trajectory.py`.

## Layout

| file | what |
|---|---|
| `pi0_policy_server.py` | CLI + gRPC server entry point; start here |
| `pi0_policy_wrapper.py` | observation/action conversion, chunk scheduling, ensembling |
| `action_blend.py` | SO(3)-correct blending of 20-d `xyzrot6g` actions |
| `episode_recorder.py` | per-episode videos + state trajectory |
| `analyze_trajectory.py` | offline plots + the chunk-seam report |
| `test_pi0_bridge_smoke.py` | full round trip against a mock policy; no GPU needed |

See `pi0_policy_server.py`'s module docstring for the full flag reference and
`pi0_policy_wrapper.py`'s for the conversion logic and its ASSUMPTIONS list.
