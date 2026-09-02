#!/usr/bin/env python3
"""Pi0.5 (openpi) policy server: bridges a running openpi `serve_policy.py`
WebSocket server to anzu's gRPC rollout client (`--mode=rollout_grpc` /
`DiffusionHwToGrpc`), the same way `diffusion_policy_server.py` and
`vla_policy_server.py` in `lbm/grpc_workspace` bridge their own backends.

See `pi0_policy_wrapper.py`'s module docstring for the conversion logic and
its documented ASSUMPTIONS (state-vector joint ordering, camera mapping) --
read those before deploying against a real checkpoint.

This process does NOT load the model itself, so it needs almost none of
openpi's dependencies. Start openpi's own server first, from your fork
checkout:

    cd <your-openpi-fork-checkout>
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config=pi05_clean_spill --policy.dir=<ckpt-dir>
    # wait for it to report it's listening (default port 8000)

...then, using the openpi fork's own venv (it already has openpi_client;
PYTHONPATH adds lbm's prismatic/diffusion_policy for a couple of pure-numpy
math helpers -- see pi0_policy_wrapper.py's imports; this directory itself
does NOT need to be on PYTHONPATH, since running the script by path already
puts it on sys.path[0], which is what makes the flat `from
pi0_policy_wrapper import ...` imports here resolve):

    cd ~/tennyyin/openpi
    # policy_interfaces must be THIS exact wheel, not a generic PyPI
    # install -- it's what makes this server's gRPC proto match what
    # anzu's client actually has loaded. Reusing lbm/grpc_workspace's own
    # proto instead (an earlier version of this file did) 100%-reproducibly
    # fails with `UNIMPLEMENTED: Method not found!`: that copy's .proto
    # declares `package lbm_policy_interface`, one export cycle behind
    # anzu's `policy_interface` (no lbm_ prefix) -- gRPC routes on the
    # full package.Service/Method string, so it's a silent version-skew,
    # not a typo. Confirmed byte-identical otherwise (see
    # pi0_policy_wrapper.py's docstring for the diff).
    uv pip install /home/robot-lab/multiarm/anzu/tools/workspace/venv/wheels/policy_interfaces-0.1.4-py3-none-any.whl
    # --no-sync: plain `uv run` re-syncs the venv against openpi's own
    # pyproject.toml/lockfile on every invocation, which would silently
    # undo the uv pip install above (it did, the first time -- downgraded
    # protobuf back under policy_interfaces' generated _pb2.py files and
    # broke them with `cannot import name 'runtime_version'`).
    PYTHONPATH=/home/robot-lab/lbm \\
      uv run --no-sync python examples/pi0_grpc_bridge/pi0_policy_server.py \\
        --checkpoint-path <ckpt-dir-or-s3-path>  \\
        --skill-type BimanualCleanUpSpill \\
        --websocket-port 8000
    # wait for "Started Server loop on localhost:50051..." -- same signal
    # as the VLA Foundry server.

--base-camera/--left-wrist-camera/--right-wrist-camera default to
scene_right_0 / wrist_left_plus / wrist_right_plus -- confirmed by
tennyyyin on 2026-09-01 against the running davis station as the mapping
for this checkpoint. Override them only if serving a different checkpoint
trained on a different camera layout.

Then run anzu's rollout exactly as for any other gRPC-served policy (see
the station's own rollout notes for the canonical invocation):

    cd ~/multiarm/anzu
    AWS_PROFILE=sagemaker ./run --build "" \\
        //intuitive/visuomotor/demo:quick_run_visuomotor_experiment \\
        --skill BimanualCleanUpSpill --station davis \\
        --demonstration_indices 0:100 --mode=rollout_grpc \\
        --policy_type=diffusion --save_async --operator_name "test" \\
        --language_instruction "pick up the knocked over cup, set it \\
upright, and wipe up the spilled liquid with a towel"

(--policy_type=diffusion is anzu's client-side flag and is unrelated to the
policy actually being pi0.5 on the server side; anzu just needs to know it's
talking rollout_grpc. `BimanualCleanUpSpill` does NOT need adding to
intuitive/skill_types/davis.txt -- the skill enum is global across every
station's *.txt, so its one existing entry in ruggles.txt already makes it
valid everywhere, `--station davis` included. Adding it a second time
throws `ValueError: Duplicates in skill type lists` -- learned this the
hard way on 2026-09-01, reverted.)

The anzu command above needs NO extra flags for data capture: every episode
is recorded server-side by this process (see episode_recorder.py) to

    ~/tennyyin/data/<checkpoint-basename>_<task-shorthand>/<timestamp>/

as one .mp4 per model-input camera view plus states.csv / states.jsonl
(commanded and reached, one row per control tick) and meta.json. An episode
is closed and written when anzu resets for the next demonstration index, or
after --record-idle-timeout-s seconds without a step -- so the data lands on
disk seconds after a run stops, whether or not the anzu side gets as far as
saving its own episode. Pass --no-record to turn it off.

One anzu-side prerequisite for the rubric ("success/failure") GUI: the skill
needs ~/efs/data/tasks/<skill>/eval_info/<skill>_rubric.txt to exist.
`quick_run_visuomotor_experiment` always passes that path, and
`preload_rubric_gui` only WARNS when it's missing -- then
`fill_rubric_via_gui` is still called (the path string is non-None) and
dies at the end of the first episode with `AttributeError: 'NoneType'
object has no attribute 'set_ready_to_be_read'`. Created for
BimanualCleanUpSpill on 2026-09-01 from BimanualTemplate_rubric.txt; add
task-specific Y/N lines to it as desired (see
BimanualUprightMugAndPlaceOnSaucer_rubric.txt for the style).

Not wired into lbm's bazel build (`BUILD.bazel`) -- this lives outside the
lbm tree and hasn't been added as a bazel py_binary. Run it directly via the
venv as shown above.
"""

import argparse
import os
import time

from pi0_policy_wrapper import DEFAULT_CLIENT_TIMEOUT_SECS
from pi0_policy_wrapper import Pi0PolicyWrapperConfig
from policy_interfaces.grpc_interface.policy_server import LbmPolicyServerConfig
from policy_interfaces.grpc_interface.policy_server import run_policy_server


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help=(
            "Path/URI of the checkpoint being served, recorded in "
            "PolicyMetadata only -- this process does not load it, "
            "serve_policy.py already has."
        ),
    )
    parser.add_argument(
        "--policy-config-name",
        type=str,
        default="pi05_clean_spill",
        help=(
            "The --policy.config value serve_policy.py was started with. "
            "Per the HF README, use this for ALL FOUR clean-spill "
            "checkpoints -- it selects model architecture/obs packing, not "
            "training recipe."
        ),
    )
    parser.add_argument(
        "--skill-type",
        type=str,
        required=True,
        help=(
            "anzu SkillType this checkpoint serves, e.g. "
            "BimanualCleanUpSpill. Must already exist in exactly one "
            "intuitive/skill_types/*.txt on the anzu checkout being used."
        ),
    )
    parser.add_argument(
        "--base-camera",
        type=str,
        default="scene_right_0",
        help=(
            "Semantic camera name (a SpartanCameraNames member, "
            "case-insensitive) feeding observation/image. Default "
            "confirmed by tennyyyin on 2026-09-01 against the running "
            "davis station for this checkpoint."
        ),
    )
    parser.add_argument(
        "--left-wrist-camera",
        type=str,
        default="wrist_left_plus",
        help=("Semantic camera name feeding observation/left_wrist_image. See --base-camera for default provenance."),
    )
    parser.add_argument(
        "--right-wrist-camera",
        type=str,
        default="wrist_right_plus",
        help=("Semantic camera name feeding observation/right_wrist_image. See --base-camera for default provenance."),
    )
    parser.add_argument(
        "--websocket-host",
        type=str,
        default="localhost",
        help="Host where serve_policy.py is listening.",
    )
    parser.add_argument(
        "--websocket-port",
        type=int,
        default=8000,
        help="Port where serve_policy.py is listening (its own default).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Only needed if serve_policy.py was started with one.",
    )
    parser.add_argument(
        "--num-open-loop-steps",
        type=int,
        default=8,
        help=(
            "Actions to execute per inference call before re-querying the "
            "policy. Must be <= the checkpoint's action_horizon (16 for "
            "pi05_clean_spill). VLA Foundry on this station uses 8 as its "
            "default; pick a value to match, or start there."
        ),
    )
    parser.add_argument(
        "--client-timeout-s",
        dest="client_timeout_s",
        type=float,
        default=DEFAULT_CLIENT_TIMEOUT_SECS,
        help=(
            "Seconds a client is remembered after its last update before "
            "its policy state is cleared. Defaults to one hour, matching "
            "the diffusion/VLA servers."
        ),
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help=(
            "Where to write one JSON line per step() (measured vs. "
            "commanded pose/gripper each tick, tagged with which "
            "inference call it came from) -- for offline analysis of a "
            "rollout, e.g. plotting height over time against chunk "
            "boundaries. Defaults to a fresh timestamped file under "
            "trajectory_logs/ next to this script (gitignored)."
        ),
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable trajectory logging entirely (--log-path is ignored).",
    )
    parser.add_argument(
        "--samples-per-call",
        type=int,
        default=4,
        help=(
            "Average this many independent draws per infer() call. "
            "Mitigates pi0.5's per-call sampling variance landing directly "
            "on the commanded pose (this checkpoint emits an absolute "
            "xyzrot6g pose, not a delta) -- root-caused and fixed on a "
            "sibling project, see action_blend.py. 1 disables averaging "
            "(this bridge's pre-fix behavior). Default 4 matches the "
            "proven config from that project."
        ),
    )
    parser.add_argument(
        "--ensemble-decay",
        type=float,
        default=0.3,
        help=(
            "ACT-style temporal ensembling: keep every chunk alive for its "
            "full predicted horizon and command the exp(-decay*age)-"
            "weighted average of every live chunk covering the current "
            "step, instead of discarding old chunks at each re-plan. "
            "Smooths the chunk-boundary seam on top of what "
            "--samples-per-call already removes. Default 0.3 matches the "
            "proven config; pass --no-ensemble to disable."
        ),
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable temporal ensembling (--ensemble-decay is ignored); "
        "falls back to the simpler discard-and-replace tape.",
    )
    parser.add_argument(
        "--ensemble-gripper",
        action="store_true",
        help=(
            "Blend/vote gripper commands the same as pose, instead of "
            "always trusting a single source (the freshest live chunk in "
            "ensemble mode, the majority vote in --samples-per-call). "
            "Off by default and it matters: recorded gripper commands are "
            "effectively binary, and blending draws that disagree about "
            "grasp phase can emit a command that never crosses the close "
            "threshold -- this cost real grasps on the sibling project "
            "before they turned it off."
        ),
    )
    parser.add_argument(
        "--prefetch-lead-ticks",
        type=int,
        default=5,
        help=(
            "How many ticks before a chunk is needed to start inferring it. "
            "Must cover one multi-draw inference (~87ms per draw) or the "
            "fresh tick stalls the control loop; must also leave "
            "prefetch-lead + num-open-loop-steps - 1 + action-latency-ticks "
            "rows inside the 16-row horizon, because a chunk arriving N "
            "ticks late is entered at row N (its first N rows are setpoints "
            "for time already past)."
        ),
    )
    parser.add_argument(
        "--action-latency-ticks",
        type=int,
        default=0,
        help=(
            "Extra rows to skip ahead, on top of the exact prefetch-"
            "staleness compensation that always happens. Raise this only if "
            "a rollout still shows fresh-chunk commands landing BEHIND the "
            "measured pose (analyze_trajectory.py reports it); 0 assumes a "
            "chunk's row 0 is the setpoint at its observation's tick."
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="/home/robot-lab/tennyyin/data",
        help=(
            "Root for per-episode rollout recordings: one directory per "
            "run at <record-dir>/<policy>_<task>/<timestamp>/ holding a "
            "video per model-input camera view plus states.csv/.jsonl "
            "(commanded and reached) and meta.json. See "
            "episode_recorder.py. Pass --no-record to disable."
        ),
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable per-episode rollout recording (--record-dir ignored).",
    )
    parser.add_argument(
        "--policy-name",
        type=str,
        default=None,
        help=(
            "First half of the recording folder name. Defaults to the last "
            "path component of --checkpoint-path (e.g. `base`, `r2_1750`), "
            "which is what distinguishes the four clean-spill checkpoints."
        ),
    )
    parser.add_argument(
        "--task-shorthand",
        type=str,
        default=None,
        help=(
            "Second half of the recording folder name. Defaults to "
            "--skill-type with any Bimanual prefix dropped and CamelCase "
            "snake_cased (BimanualCleanUpSpill -> clean_up_spill)."
        ),
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=10.0,
        help=(
            "Nominal playback rate of the recorded videos. Should match the "
            "station's control rate (~10Hz on davis) so playback is "
            "real-time; the true per-frame timestamps are in states.csv "
            "either way."
        ),
    )
    parser.add_argument(
        "--record-max-width",
        type=int,
        default=640,
        help=(
            "Downscale recorded frames to at most this width (0 = keep "
            "native resolution). The model itself sees 224x224, so 640 is "
            "already generous for eyeballing an observation; native "
            "Blackfly frames make the files several times larger for no "
            "diagnostic gain."
        ),
    )
    parser.add_argument(
        "--record-idle-timeout-s",
        type=float,
        default=5.0,
        help=(
            "Close and save an episode after this many seconds with no "
            "step. This is what saves the data right after the operator "
            "marks success/failure: anzu then blocks on its rubric GUI and "
            "may not reset for a long time, and waiting for that reset "
            "would leave the episode unflushed."
        ),
    )
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    args = parser.parse_args()

    log_path = None
    if not args.no_log:
        log_path = args.log_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "trajectory_logs",
            f"{time.strftime('%Y%m%dT%H%M%S')}_{args.skill_type}.jsonl",
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"Logging trajectory to {log_path}")

    policy_config = Pi0PolicyWrapperConfig(
        websocket_host=args.websocket_host,
        websocket_port=args.websocket_port,
        api_key=args.api_key,
        checkpoint_path=args.checkpoint_path,
        policy_config_name=args.policy_config_name,
        base_camera=args.base_camera,
        left_wrist_camera=args.left_wrist_camera,
        right_wrist_camera=args.right_wrist_camera,
        skill_type=args.skill_type,
        num_open_loop_steps=args.num_open_loop_steps,
        client_timeout_s=args.client_timeout_s,
        batch=True,
        log_path=log_path,
        samples_per_call=args.samples_per_call,
        ensemble_decay=None if args.no_ensemble else args.ensemble_decay,
        ensemble_gripper=args.ensemble_gripper,
        prefetch_lead_ticks=args.prefetch_lead_ticks,
        action_latency_ticks=args.action_latency_ticks,
        record_dir=None if args.no_record else args.record_dir,
        policy_name=args.policy_name,
        task_shorthand=args.task_shorthand,
        record_fps=args.record_fps,
        record_max_width=args.record_max_width,
        record_idle_timeout_s=args.record_idle_timeout_s,
    )
    policy = policy_config.create()
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
