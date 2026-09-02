"""gRPC-served wrapper around an openpi Pi0.5 policy.

Bridges anzu's `rollout_grpc` client (the same `LbmPolicyServer` protocol used
for the diffusion-policy and VLA-Foundry backends -- see
`lbm/grpc_workspace/diffusion_policy_server.py` and
`.../vla_policy_server.py`) to a running openpi `serve_policy.py` WebSocket
server. Nothing on the anzu/davis side changes: davis only needs *something*
speaking gRPC on :50051, and this process is that something.

Two processes, mirroring the VLA Foundry two-process pattern:

  1. `serve_policy.py` (from your openpi fork) -- loads the checkpoint and
     serves it over WebSocket. Not part of this repo; run it separately:

         cd <openpi-fork-checkout>
         GIT_LFS_SKIP_SMUDGE=1 uv sync
         uv run scripts/serve_policy.py policy:checkpoint \
             --policy.config=pi05_clean_spill --policy.dir=<ckpt-dir>

  2. `pi0_policy_server.py` (this directory) -- connects to (1) as a
     WebSocket client via `openpi_client.websocket_client_policy
     .WebsocketClientPolicy`, and re-serves it over gRPC on :50051 using
     the `LbmPolicyServer` framework -- but from the `policy_interfaces`
     PyPI-style wheel anzu itself depends on
     (`~/multiarm/anzu/tools/workspace/venv/wheels/policy_interfaces-*.whl`),
     NOT `lbm/grpc_workspace`'s copy of the same code. The two are
     near-identical (this module still gets `Policy`/`MultiarmObservation`/
     `PosesAndGrippers` from plain `robot_gym`, confirmed byte-for-byte
     identical to `policy_interfaces.robot_gym`), but their compiled
     `.proto` differ: `lbm`'s declares `package lbm_policy_interface`,
     while anzu's loaded `policy_interfaces` declares `package
     policy_interface` (no `lbm_` prefix). gRPC routes by the full
     `package.Service/Method` string, so serving off `lbm`'s copy is a
     silent version-skew that anzu's client sees as
     `UNIMPLEMENTED: Method not found!` on the very first RPC
     (`GetPolicyMetadata`) -- confirmed by hitting exactly that on
     2026-09-01. Only `pi0_policy_server.py`'s import needs to point at the
     right one; nothing here in the wrapper cares which copy served it.

Why this lives outside the shared `~/lbm` checkout: that checkout has no
venv provisioned yet and is the lab's shared, git-tracked copy (same reason
PRO-DAgger vendors its own copy of anzu rather than patching
`~/multiarm/anzu` directly). This module only *imports* from `lbm` (via
PYTHONPATH -- see `pi0_policy_server.py`'s docstring); it does not modify it.

CONFIRMED (2026-09-01, against `~/tennyyin/openpi/examples/bike_rotor/
convert_bike_rotor_to_lerobot.py`, the actual training-data-generation source
-- not the HF README, which doesn't state this) -- `observation/state` is
`[left_joint_0..6, right_joint_0..6, left_gripper, right_gripper]`
(**left-first**), while the 20-d **action** is right-first (see below).
State and action do NOT share an arm-order convention; do not assume one
from the other. Source comment:

    observation.state = actual joint_position_left(7) + joint_position_right(7)
                        + gripper_left(1) + gripper_right(1)

This was wrong in an earlier version of this file (right-first, "inferred"
from the action order rather than confirmed) -- that produced exactly the
failure this class of bug predicts: garbled proprioception, not a crash, and
visibly nonsensical actions on real hardware. Root-caused on 2026-09-01 by
reading the conversion script rather than continuing to guess; see
`test_pi0_bridge_smoke.py` for a regression test pinning this ordering.

ASSUMPTIONS remaining -- verify these once more real-world rollouts are
available. Getting any of them wrong fails silently (a badly-formed
observation, not a crash) and degrades the policy rather than erroring
loudly, so they're called out explicitly rather than buried in code:

  * Actions are already ABSOLUTE poses (per the HF README AND confirmed by
    `bike_rotor_policy.py`'s docstring: "absolute end-effector targets...
    We do NOT apply a delta-to-state transform"), so unlike
    `VLAPolicyWrapper._format_action` in
    `lbm/prismatic/vla/deploy/vla_policy_wrapper.py`, there is no
    relative-to-absolute conversion step here.
Also CONFIRMED, not assumed (both against `ModelTransformFactory` in
`~/tennyyin/openpi/src/openpi/training/config.py`, applied server-side by
`serve_policy.py` for every request, so nothing below needs client-side
handling):

  * Images: sent at native camera resolution/dtype (uint8, HxWx3) is
    correct as-is -- `_transforms.ResizeImages(224, 224)` runs server-side
    for `ModelType.PI05`. Client-side resizing would be redundant, not
    wrong, but there's no need for it.
  * State/action padding to the model's internal dims: handled server-side
    by `_transforms.PadStatesAndActions(model_config.action_dim)`.
  * Normalization: handled server-side using the checkpoint's own
    `norm_stats` (loaded from `assets/`, logged at server startup as
    `Loaded norm stats from .../assets/tri/clean_spill_cartesian`) -- the
    client sends/receives raw, unnormalized values throughout.
  * Camera mapping: CONFIRMED twice over -- first by tennyyyin against the
    running station (2026-09-01), then independently by
    `bike_rotor_policy.py`'s own docstring (`base_0_rgb <- scene_right_0`,
    `left_wrist_0_rgb <- wrist_left_plus`, `right_wrist_0_rgb <-
    wrist_right_plus`) -- as `base_camera=scene_right_0`,
    `left_wrist_camera=wrist_left_plus`, `right_wrist_camera=wrist_right_plus`,
    now the defaults in `Pi0PolicyWrapperConfig` below.

Nothing in this module's ASSUMPTIONS/CONFIRMED lists remains unverified as
of 2026-09-01.
"""

import concurrent.futures
import dataclasses as dc
import json
import threading
import time
import uuid

from action_blend import GRIP_COLS
from action_blend import blend_actions20
from diffusion_policy.common.pose_util import rot6d_to_mat
import numpy as np
from prismatic.vla.deploy.ros_robot_policy_conversions import array_to_rotation_matrix
from prismatic.vla.deploy.ros_robot_policy_conversions import np_shaped_3x3_array_to_flat_array
from prismatic.vla.deploy.vla_observation_config import SpartanCameraNames
from pydrake.math import RigidTransform
from robot_gym.multiarm_spaces import MultiarmObservation
from robot_gym.multiarm_spaces import PosesAndGrippers
from robot_gym.policy import Policy
from robot_gym.policy import PolicyConfig
from robot_gym.policy import PolicyMetadata

# Define the default client timeout to one hour in seconds, matching every
# other *_policy_wrapper.py in lbm (diffusion_policy_wrapper.py,
# vla_policy_wrapper.py).
DEFAULT_CLIENT_TIMEOUT_SECS = 3600.0

# anzu's model/gripper names for a bimanual dual-Panda station. Same
# constants as every other policy wrapper in lbm
# (robot_gym/multiarm_spaces.py, RestorePosesAndGrippersConfig.make_default).
_RIGHT_ARM = "right::panda"
_LEFT_ARM = "left::panda"
_RIGHT_GRIPPER = "right::panda_hand"
_LEFT_GRIPPER = "left::panda_hand"

# [T, 20] action layout per the HF README:
# right XYZ(0:3) | right 6D rotation(3:9) | left XYZ(9:12) |
# left 6D rotation(12:18) | right gripper(18) | left gripper(19)
_ACTION_DIM = 20
_ACTION_HORIZON = 16  # pi05_clean_spill's action_horizon.
_STATE_DIM = 16  # 14 joints + 2 grippers.


class Pi0PolicyWrapperBase(Policy):
    """Shared observation/action conversion; see the module docstring."""

    def __init__(
        self,
        *,
        websocket_host: str,
        websocket_port: int,
        api_key: str | None,
        checkpoint_path: str,
        policy_config_name: str,
        base_camera: str,
        left_wrist_camera: str,
        right_wrist_camera: str,
        skill_type: str,
        num_open_loop_steps: int,
        samples_per_call: int = 1,
        ensemble_decay: float | None = None,
        ensemble_gripper: bool = False,
        action_latency_ticks: int = 0,
        prefetch_lead_ticks: int = 5,
        log_path: str | None = None,
        record_dir: str | None = None,
        task_shorthand: str | None = None,
        policy_name: str | None = None,
        record_fps: float = 10.0,
        record_max_width: int = 640,
        record_idle_timeout_s: float = 5.0,
    ):
        super().__init__()

        # Deferred import: `openpi_client` is a small, JAX-free package
        # (NOT the full openpi/JAX training stack) that is not part of the
        # shared lbm venv's requirements.in. See pi0_policy_server.py's
        # docstring for the one-line `pip install openpi-client` needed to
        # add it. Deferring the import keeps this module importable (e.g.
        # for tests) even before that's installed.
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=websocket_host, port=websocket_port, api_key=api_key)
        # WebsocketClientPolicy holds one live socket and is not documented
        # as thread-safe (a bare send()/recv() pair, no request IDs to
        # de-multiplex responses). Pi0PolicyWrapperBatch runs inference in a
        # background thread to prefetch the next action chunk (see its
        # docstring) -- this lock keeps that safe even if it ever overlaps
        # with a synchronous call for a second client, by serializing
        # rather than truly parallelizing concurrent infer() calls.
        self._infer_lock = threading.Lock()
        self._checkpoint_path = checkpoint_path
        self._policy_config_name = policy_config_name
        self._skill_type = skill_type
        assert num_open_loop_steps > 0
        assert num_open_loop_steps <= _ACTION_HORIZON, (
            f"num_open_loop_steps ({num_open_loop_steps}) must be <= this "
            f"checkpoint's action_horizon ({_ACTION_HORIZON})"
        )
        self._num_open_loop_steps = num_open_loop_steps

        assert samples_per_call >= 1
        self._samples_per_call = samples_per_call
        assert ensemble_decay is None or ensemble_decay >= 0.0
        self._ensemble_decay = ensemble_decay
        self._ensemble_gripper = ensemble_gripper
        # Extra rows to skip ahead, ON TOP OF the exact prefetch-staleness
        # compensation the step loops already do. That compensation lines a
        # chunk's rows up with the tick each row was predicted FOR; this
        # knob exists because a residual offset can survive it -- the
        # command still has to travel to the controller and be tracked, and
        # a chunk's row 0 is (per training-data convention) the setpoint AT
        # the observation's tick, not one tick after it. Measured on real
        # hardware 2026-09-01: even with only 1 tick of prefetch staleness,
        # a fresh chunk's row 0 landed ~9mm BEHIND the measured pose while
        # mid-chunk commands ran ~16mm AHEAD of it. Leave at 0 and re-measure
        # first (see analyze_trajectory.py) -- the staleness fix alone may
        # account for all of it.
        assert action_latency_ticks >= 0
        self._action_latency_ticks = action_latency_ticks

        # How many ticks before a chunk is needed to start inferring it. Must
        # cover one _infer_chunk_blended (samples_per_call serialized draws,
        # ~87ms each) or the fresh tick blocks on future.result() and stalls
        # the control loop -- measured at 0.348s vs a 0.093s normal tick when
        # 4 draws had only 1 tick of lead. But lead is not free either: every
        # lead tick is a chunk row spent on the past (see the fresh-chunk
        # branches), so lead + num_open_loop_steps - 1 + action_latency_ticks
        # rows of the horizon must exist. 5 covers 4 draws with slack and
        # still leaves headroom at num_open_loop_steps=8.
        assert prefetch_lead_ticks >= 1
        assert prefetch_lead_ticks <= num_open_loop_steps, (
            f"prefetch_lead_ticks ({prefetch_lead_ticks}) cannot exceed "
            f"num_open_loop_steps ({num_open_loop_steps}) -- a chunk would "
            "have to be requested before the previous one started."
        )
        self._prefetch_lead_ticks = prefetch_lead_ticks
        # Fail loudly here rather than as a mid-rollout RuntimeError from
        # _step_batch_ensembled's "no live chunk covers step N".
        deepest_row = prefetch_lead_ticks + num_open_loop_steps - 1 + action_latency_ticks
        assert deepest_row < _ACTION_HORIZON, (
            f"prefetch_lead_ticks({prefetch_lead_ticks}) + "
            f"num_open_loop_steps({num_open_loop_steps}) - 1 + "
            f"action_latency_ticks({action_latency_ticks}) = {deepest_row} "
            f"needs row {deepest_row} of a {_ACTION_HORIZON}-row chunk. "
            "Lower the lead, the cadence, or the latency."
        )

        # Raises a clear KeyError immediately if a bad camera name was
        # passed, rather than failing later mid-rollout.
        self._base_camera = SpartanCameraNames[base_camera.upper()]
        self._left_wrist_camera = SpartanCameraNames[left_wrist_camera.upper()]
        self._right_wrist_camera = SpartanCameraNames[right_wrist_camera.upper()]
        # Model-input slot -> the camera actually feeding it, used to name
        # the recorded video files. The camera name is in the filename on
        # purpose: a mis-mapped or dead wrist camera is then obvious from
        # `ls` plus one look at the video, which is the whole reason
        # per-view recording exists (2026-09-01: one arm performing much
        # worse than the other, suspected bad observation).
        self._view_cameras = {
            "base": self._base_camera.name.lower(),
            "left_wrist": self._left_wrist_camera.name.lower(),
            "right_wrist": self._right_wrist_camera.name.lower(),
        }

        # Trajectory logging -- one JSON line per step(), so a rollout can
        # be inspected offline (e.g. plot measured vs. commanded height
        # over time) instead of debugging jerkiness/oscillation purely by
        # description. See _log_step() and pi0_policy_server.py's
        # --log-path.
        self._log_lock = threading.Lock()
        # One append handle for the whole server lifetime, so a context
        # manager can't own it (hence the SIM115 waiver).
        self._log_file = open(log_path, "a") if log_path else None  # noqa: SIM115

        # Per-episode rollout recording (3 videos + commanded/reached state
        # trajectory per run) -- see episode_recorder.py. Separate from
        # --log-path above, which is one long append-only file across every
        # episode for trajectory analysis; this one is per-run archival
        # data, indexed by policy and task.
        self._step_counts: dict[uuid.UUID, int] = {}
        self._recorder = None
        self._record_enabled = False
        if record_dir:
            from episode_recorder import RolloutRecorder
            from episode_recorder import default_policy_name
            from episode_recorder import default_task_shorthand

            self._recorder = RolloutRecorder(
                root_dir=record_dir,
                policy_name=policy_name or default_policy_name(checkpoint_path),
                task_shorthand=(task_shorthand or default_task_shorthand(skill_type)),
                state_dim=_STATE_DIM,
                fps=record_fps,
                max_width=record_max_width,
                idle_timeout_s=record_idle_timeout_s,
                base_meta={
                    "checkpoint_path": checkpoint_path,
                    "policy_config_name": policy_config_name,
                    "skill_type": skill_type,
                    "num_open_loop_steps": num_open_loop_steps,
                    "samples_per_call": samples_per_call,
                    "ensemble_decay": ensemble_decay,
                    "ensemble_gripper": ensemble_gripper,
                    "prefetch_lead_ticks": prefetch_lead_ticks,
                    "action_latency_ticks": action_latency_ticks,
                    "trajectory_log_path": log_path,
                },
            )
            self._record_enabled = True
            # A rollout can be stopped with Ctrl-C on the anzu side and this
            # server killed afterwards; flush whatever episode is open
            # rather than losing it. (The idle watchdog covers the normal
            # case -- see RolloutRecorder.)
            import atexit

            atexit.register(self._recorder.close)

    def _record_step(
        self,
        client_uuid: uuid.UUID,
        record: dict,
        observation: MultiarmObservation,
    ) -> None:
        """Hand one tick to the episode recorder. Runs on the control-loop
        thread, so it must never raise: a recording problem (e.g. a camera
        missing from `visuo`) must not drop a moving robot. Frame *encoding*
        happens on the recorder's own thread."""
        if not self._record_enabled:
            return
        try:
            images = self._extract_three_images(observation)
            self._recorder.record_step(
                client_uuid,
                record,
                {
                    "base": images["observation/image"],
                    "left_wrist": images["observation/left_wrist_image"],
                    "right_wrist": images["observation/right_wrist_image"],
                },
                self._view_cameras,
                extra_meta={
                    "language_instruction": observation.language_instruction,
                },
            )
        except Exception:
            import traceback

            # Disabled rather than retried: every plausible cause here is a
            # persistent config mismatch, so retrying would just reprint
            # this every 100ms for the rest of the rollout. Already-open
            # episodes still flush (the recorder's idle watchdog owns that).
            self._record_enabled = False
            print(
                "[record] frame capture failed -- episode recording "
                "DISABLED for the rest of this server's life (the rollout "
                "itself is unaffected):\n" + traceback.format_exc()
            )

    def _log_step(
        self,
        *,
        client_uuid: uuid.UUID,
        chunk_id: int,
        is_fresh_chunk: bool,
        step_within_chunk: int,
        observation: MultiarmObservation,
        sent_state: np.ndarray | None,
        action: PosesAndGrippers,
    ) -> None:
        """Build this tick's record and hand it to both sinks: the
        append-across-episodes trajectory log (--log-path) and the
        per-episode recorder (--record-dir). Both are optional; the record
        is built once either way so the two can never disagree."""
        if self._log_file is None and not self._record_enabled:
            return
        actual = observation.robot.actual
        step = self._step_counts.get(client_uuid, 0)
        self._step_counts[client_uuid] = step + 1

        def _block(poses, grippers, joints):
            """One arm-pair's pose/quaternion/gripper (and joints, when
            available) in the flat `<side>_<field>` shape both sinks read.
            `*_pos` keys are unchanged from the original log format --
            analyze_trajectory.py still reads them."""
            out = {}
            for side, arm, gripper in (
                ("right", _RIGHT_ARM, _RIGHT_GRIPPER),
                ("left", _LEFT_ARM, _LEFT_GRIPPER),
            ):
                pose = poses[arm]
                out[f"{side}_pos"] = [float(v) for v in pose.translation()]
                out[f"{side}_quat"] = [float(v) for v in pose.rotation().ToQuaternion().wxyz()]
                out[f"{side}_gripper"] = float(grippers[gripper])
                if joints is not None and arm in joints:
                    out[f"{side}_joints"] = [float(v) for v in np.asarray(joints[arm]).ravel()]
            return out

        record = {
            "t": time.time(),
            "client": str(client_uuid),
            "step": step,
            "chunk_id": chunk_id,
            "step_within_chunk": step_within_chunk,
            "is_fresh_chunk": is_fresh_chunk,
            # What was actually measured on the robot this tick (ground
            # truth, independent of whether this tick called infer()) --
            # the "reached" half. Includes joint angles, which are the
            # station's primary measurement; the poses are FK of these.
            "measured": _block(actual.poses, actual.grippers, actual.joint_position),
            # The 16-d vector actually sent to the model -- only present
            # on ticks that called infer() (fresh chunks); None otherwise,
            # since no new observation/state was built for a tape-pop.
            "sent_state": sent_state.tolist() if sent_state is not None else None,
            # What this tick is commanding the robot to do -- the
            # "commanded" half. Absolute end-effector poses, no joint
            # targets: that is all this checkpoint emits.
            "commanded": _block(action.poses, action.grippers, None),
        }
        if self._log_file is not None:
            line = json.dumps(record)
            with self._log_lock:
                self._log_file.write(line + "\n")
                self._log_file.flush()
        self._record_step(client_uuid, record, observation)

    def _extract_three_images(self, observation: MultiarmObservation) -> dict[str, np.ndarray]:
        """Pull exactly the 3 cameras this checkpoint expects out of
        `observation.visuo`, which is keyed by the semantic camera name
        (see `VLAPolicyWrapperBase._extract_images_from_observation` in
        `lbm/prismatic/vla/deploy/vla_policy_wrapper.py` for the same
        lookup pattern used elsewhere in lbm).

        Raises rather than silently substituting a blank image (unlike the
        VLA wrapper): a configured camera being absent means the station's
        actual camera list doesn't match `--base-camera`/
        `--left-wrist-camera`/`--right-wrist-camera`, which should fail
        loudly rather than quietly hand the model a black frame.
        """
        semantic_to_array = {}
        for camera_key, image_set in observation.visuo.items():
            assert camera_key.upper() in SpartanCameraNames.__members__, (
                f"camera key {camera_key!r} is not a known SpartanCameraNames entry"
            )
            semantic_to_array[SpartanCameraNames[camera_key.upper()]] = image_set.rgb.array

        def _get(camera: SpartanCameraNames, slot_name: str) -> np.ndarray:
            if camera not in semantic_to_array:
                raise RuntimeError(
                    f"Configured {slot_name} camera {camera.name} was not "
                    "present in this observation. Cameras seen this step: "
                    f"{sorted(c.name for c in semantic_to_array)}. Check "
                    "--base-camera/--left-wrist-camera/--right-wrist-camera "
                    "against the station's actual camera list."
                )
            return semantic_to_array[camera]

        return {
            "observation/image": _get(self._base_camera, "base"),
            "observation/left_wrist_image": _get(self._left_wrist_camera, "left_wrist"),
            "observation/right_wrist_image": _get(self._right_wrist_camera, "right_wrist"),
        }

    def _build_state(self, observation: MultiarmObservation) -> np.ndarray:
        """Left-first: [left_joints(7), right_joints(7), left_gripper,
        right_gripper]. Confirmed against convert_bike_rotor_to_lerobot.py
        -- see the module docstring. Note this is the OPPOSITE arm-order
        from the (right-first) action layout in _action_chunk_to_tape.
        """
        actual = observation.robot.actual
        for arm in (_RIGHT_ARM, _LEFT_ARM):
            if actual.joint_position is None or arm not in actual.joint_position:
                raise RuntimeError(
                    f"observation.robot.actual.joint_position[{arm!r}] is "
                    "missing. This checkpoint needs joint-space "
                    "proprioception (14 joints + 2 grippers), not just "
                    "poses -- check that the station's robot bridge is "
                    "populating joint_position (it's an optional field on "
                    "PosesAndGrippers)."
                )
        right_joints = np.asarray(actual.joint_position[_RIGHT_ARM], dtype=np.float32)
        left_joints = np.asarray(actual.joint_position[_LEFT_ARM], dtype=np.float32)
        shapes = f"{right_joints.shape} / {left_joints.shape}"
        assert right_joints.shape == (7,), f"expected 7 joints per arm, got {shapes}"
        assert left_joints.shape == (7,), f"expected 7 joints per arm, got {shapes}"
        right_gripper = np.float32(actual.grippers[_RIGHT_GRIPPER])
        left_gripper = np.float32(actual.grippers[_LEFT_GRIPPER])
        state = np.concatenate([left_joints, right_joints, [left_gripper], [right_gripper]]).astype(np.float32)
        assert state.shape == (_STATE_DIM,)
        return state

    def _safe_infer(self, obs_dict: dict) -> dict:
        with self._infer_lock:
            return self._client.infer(obs_dict)

    def _build_obs_dict(self, observation: MultiarmObservation) -> dict:
        if observation.language_instruction is None:
            raise RuntimeError(
                "The observation is missing a language instruction "
                "(prompt). Never invent one here -- a wrong prompt "
                "silently degrades a language-conditioned policy."
            )
        obs_dict = self._extract_three_images(observation)
        obs_dict["observation/state"] = self._build_state(observation)
        obs_dict["prompt"] = observation.language_instruction
        return obs_dict

    def _decode_action_row(self, step: np.ndarray, observation: MultiarmObservation) -> PosesAndGrippers:
        """One 20-d ABSOLUTE-pose row (see the module docstring -- no
        relative->absolute conversion needed here, unlike
        VLAPolicyWrapper._format_action) -> PosesAndGrippers.
        """
        action = PosesAndGrippers(
            poses=dict(observation.robot.actual.poses),
            grippers=dict(observation.robot.actual.grippers),
        )
        for arm_name, xyz, rot6d in (
            (_RIGHT_ARM, step[0:3], step[3:9]),
            (_LEFT_ARM, step[9:12], step[12:18]),
        ):
            rotation = array_to_rotation_matrix(np_shaped_3x3_array_to_flat_array(rot6d_to_mat(rot6d[None])[0]))
            action.poses[arm_name] = RigidTransform(p=np.asarray(xyz, dtype=np.float64), R=rotation)
        action.grippers[_RIGHT_GRIPPER] = max(float(step[18]), 0.0)
        action.grippers[_LEFT_GRIPPER] = max(float(step[19]), 0.0)
        return action

    def _action_chunk_to_tape(
        self, action_chunk: np.ndarray, observation: MultiarmObservation
    ) -> list[PosesAndGrippers]:
        """action_chunk: float32[action_horizon, 20] -> the first
        num_open_loop_steps rows, each decoded independently."""
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        expected = f"expected [T, {_ACTION_DIM}] action chunk, got {action_chunk.shape}"
        assert action_chunk.ndim == 2, expected
        assert action_chunk.shape[1] == _ACTION_DIM, expected
        n_steps = min(self._num_open_loop_steps, action_chunk.shape[0])
        return [self._decode_action_row(action_chunk[t], observation) for t in range(n_steps)]

    def _infer_chunk_blended(self, obs_dict: dict) -> np.ndarray:
        """One float32[action_horizon, 20] chunk, averaged over
        `samples_per_call` independent draws from the same obs_dict.

        Mitigates pi0.5's per-call sampling variance on this absolute
        action space -- ported 2026-09-01 from a sibling project
        (`OpenPIBikePolicy._infer_chunk`) that hit and fixed the exact same
        jerkiness; see action_blend.py's module docstring for the full
        diagnosis and citation. `samples_per_call=1` (the default) makes
        this a single plain infer() call, unchanged from before this fix.

        Gripper columns are majority-voted across the K draws, never
        blended: recorded gripper commands are effectively binary, and
        averaging draws that disagree about grasp *phase* produces a
        command that appears nowhere in training and can miss a grasp
        (see action_blend.py). `method="lower"` always returns an
        actually-sampled value and breaks an even split toward closing.
        """
        draws = [
            np.asarray(self._safe_infer(obs_dict)["actions"], dtype=np.float32) for _ in range(self._samples_per_call)
        ]
        if len(draws) == 1:
            return draws[0]
        stack = np.stack(draws)  # [K, H, 20]
        blended = np.stack([blend_actions20(stack[:, h]) for h in range(stack.shape[1])])
        if not self._ensemble_gripper:
            blended[:, GRIP_COLS] = np.percentile(stack[:, :, GRIP_COLS], 50, axis=0, method="lower")
        return blended.astype(np.float32)

    def _get_policy_metadata_impl(self, name: str) -> PolicyMetadata:
        return PolicyMetadata(
            name=name,
            skill_type=self._skill_type,
            checkpoint_path=self._checkpoint_path,
            is_language_conditioned=True,
            raw_policy_config={
                "policy_config_name": self._policy_config_name,
                "base_camera": self._base_camera.name,
                "left_wrist_camera": self._left_wrist_camera.name,
                "right_wrist_camera": self._right_wrist_camera.name,
                "num_open_loop_steps": self._num_open_loop_steps,
                "samples_per_call": self._samples_per_call,
                "ensemble_decay": self._ensemble_decay,
                "ensemble_gripper": self._ensemble_gripper,
                "action_latency_ticks": self._action_latency_ticks,
                "prefetch_lead_ticks": self._prefetch_lead_ticks,
            },
        )


class Pi0PolicyWrapper(Pi0PolicyWrapperBase):
    """Non-batch wrapper. Mirrors `VLAPolicyWrapper` /
    `DiffusionPolicyImageWrapper`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._action_tape: list[PosesAndGrippers] = []

    def reset(self, *, seed=None, options=None) -> None:
        del seed, options  # Unused: no policy-side state to seed.
        self._action_tape = []

    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        if not self._action_tape:
            obs_dict = self._build_obs_dict(observation)
            chunk = self._infer_chunk_blended(obs_dict)
            # No prefetch here, so the only staleness is the synchronous
            # inference this tick already paid for -- action_latency_ticks
            # still applies (clamped to keep the tape non-empty).
            drop = min(self._action_latency_ticks, chunk.shape[0] - 1)
            self._action_tape = self._action_chunk_to_tape(chunk[drop:], observation)
        return self._action_tape.pop(0)

    def get_policy_metadata(self):
        return self._get_policy_metadata_impl("Pi0Policy")


class Pi0PolicyWrapperBatch(Pi0PolicyWrapperBase):
    """Batch wrapper, keyed by client UUID. Mirrors
    `VLAPolicyWrapperBatch` / `DiffusionPolicyImageWrapperBatch` --
    `LbmPolicyServer` dispatches to this when `batch=True` is passed to the
    corresponding *PolicyConfig.

    Background-prefetches the next action chunk instead of calling
    `infer()` synchronously on the tick that needs it. Without this,
    `step_batch` blocks for the model's real inference latency (~150-250ms
    per the HF README) once every `num_open_loop_steps` ticks -- at 10 Hz
    control that's a stall clearly visible as periodic jerk, confirmed on
    real hardware on 2026-09-01. `DiffusionPolicyImageWrapperBatch`
    (`lbm/diffusion_policy/policy_wrapper/diffusion_policy_wrapper.py`)
    avoids the same problem with a background *process*
    (`PolicyInferenceProcess`); a background *thread* is enough here since
    our own work per call is an I/O-bound websocket round trip, not local
    compute -- the GPU work happens server-side either way.

    Mechanics: once a client's tape is down to its last queued step, kick
    off inference for the *next* chunk in a background thread using the
    observation from that same tick, then keep serving the (already
    computed) remaining tape entries while it runs. By the time the tape
    actually empties, the model call has had `(num_open_loop_steps - 1)`
    ticks of head-start -- 7 * 100ms = 700ms of margin against a ~250ms
    call, comfortably enough that the final `.result()` should never
    actually block. If it somehow hasn't finished (e.g. a slow first call,
    or this ever runs with num_open_loop_steps=1), step_batch blocks on it
    rather than firing a redundant second call.

    Note this doesn't get a real inference-time *batching* speedup:
    openpi's `WebsocketClientPolicy` serves one request per round trip, so
    multiple clients needing a fresh chunk in the same tick are still
    inferred one at a time (serialized by `_infer_lock`, not run
    concurrently) -- prefetching removes the *stall*, not the per-call
    latency. Only matters once this server is pointed at more than one
    simultaneous anzu rollout, which is not the davis setup today.

    Two stepping modes, chosen by whether `ensemble_decay` is set:

    - **Tape mode** (`ensemble_decay=None`, the default): as described
      above -- one chunk executed open-loop for `num_open_loop_steps`
      ticks, discarded, repeat. `samples_per_call` (if > 1) still applies
      within each `infer()` call via `_infer_chunk_blended`.
    - **Ensemble mode** (`ensemble_decay` set): ACT-style temporal
      ensembling, ported 2026-09-01 from `OpenPIBikePolicy._act_ensembled`
      on a sibling project that hit and fixed this bridge's exact
      jerkiness (see action_blend.py). A fresh chunk is still computed
      every `num_open_loop_steps` ticks, but old chunks are NOT discarded
      -- each stays "live" for its full predicted horizon, and every tick
      blends whichever chunks still cover it, weighted
      `exp(-ensemble_decay * age)` (freshest chunk dominates -- ~70% at
      decay=0.3 -- so the loop stays reactive while older chunks smooth
      the seam). Rotations blend on SO(3); the gripper always comes from
      the single freshest live chunk (`ensemble_gripper=False`, the
      default), never blended across chunks, for the same reason
      `_infer_chunk_blended` never blends it across draws.
    """

    def __init__(self, client_timeout_s: float = DEFAULT_CLIENT_TIMEOUT_SECS, **kwargs):
        super().__init__(**kwargs)
        # Used when this policy is called via the non-batch interface.
        self._internal_uuid = uuid.uuid4()
        self._action_tapes: dict[uuid.UUID, list] = {}
        self._known_clients: dict[uuid.UUID, float] = {}
        # Each pending prefetch keeps its Future alongside (a) the exact
        # observation/state array it was submitted with, purely so
        # _log_step can record what was actually sent once the result is
        # consumed (the state array is otherwise cheap and small; kept
        # rather than rebuilt to avoid claiming a value close to
        # (but not exactly) what was really sent, if the observation
        # were rebuilt from a slightly different MultiarmObservation), and
        # (b) HOW STALE the chunk will be when it is finally consumed --
        # the tick the observation was captured on (ensemble mode) or the
        # number of ticks that will elapse before consumption (tape mode).
        # (b) is load-bearing, not diagnostic: a chunk of ABSOLUTE poses
        # inferred N ticks ago has its first N rows already in the past, so
        # replaying it from row 0 commands the arm back to where it was N
        # ticks earlier. See _step_batch_ensembled for the measured cost.
        self._pending_infer: dict[uuid.UUID, tuple[concurrent.futures.Future, np.ndarray, int]] = {}
        # Small pool: one prefetch in flight per client is all this needs;
        # sized a little above that purely as slack, not for throughput.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="pi0-prefetch")
        assert client_timeout_s > 0.0
        self._client_timeout_s = client_timeout_s
        # For _log_step: which inference call (monotonic per client) the
        # current tape came from, and its original length (so step_within_
        # chunk can be recovered from however many items are left).
        self._chunk_ids: dict[uuid.UUID, int] = {}
        self._next_chunk_id: dict[uuid.UUID, int] = {}
        self._chunk_lengths: dict[uuid.UUID, int] = {}
        # Ensemble-mode-only state (see class docstring): every chunk still
        # inside its predicted horizon, per client, plus a per-client tick
        # counter (their "self._step") used to compute each chunk's age.
        self._live_chunks: dict[uuid.UUID, list] = {}
        self._client_step: dict[uuid.UUID, int] = {}

    def _curate_known_clients(self, recent_clients) -> None:
        now = time.time()
        for client in recent_clients:
            self._known_clients[client] = now
        stale = [
            client
            for client in self._known_clients
            if client != self._internal_uuid and now - self._known_clients[client] > self._client_timeout_s
        ]
        for client in stale:
            if self._recorder is not None:
                self._recorder.end_episode(client)
            self._action_tapes.pop(client, None)
            self._live_chunks.pop(client, None)
            self._client_step.pop(client, None)
            self._step_counts.pop(client, None)
            pending = self._pending_infer.pop(client, None)
            if pending is not None:
                pending[0].cancel()
            self._known_clients.pop(client, None)

    def step_batch(self, observations: dict[uuid.UUID, MultiarmObservation]) -> dict[uuid.UUID, PosesAndGrippers]:
        self._curate_known_clients(observations.keys())
        if self._ensemble_decay is not None:
            return self._step_batch_ensembled(observations)
        return self._step_batch_tape(observations)

    def _step_batch_tape(self, observations: dict[uuid.UUID, MultiarmObservation]) -> dict[uuid.UUID, PosesAndGrippers]:
        actions: dict[uuid.UUID, PosesAndGrippers] = {}
        for client_uuid, observation in observations.items():
            tape = self._action_tapes.get(client_uuid) or []
            is_fresh_chunk = not tape
            sent_state = None
            if is_fresh_chunk:
                # Empty tape: either this client's first call ever (no
                # prefetch could have been started), or a prefetch is
                # already in flight -- either way, get the chunk from
                # there rather than firing a second, redundant infer().
                pending = self._pending_infer.pop(client_uuid, None)
                if pending is not None:
                    future, sent_state, stale_ticks = pending
                    chunk = future.result()
                    # Latency compensation, same idea as ensemble mode's
                    # anchoring: this chunk was inferred `stale_ticks` ticks
                    # ago, so rows [0, stale_ticks) are setpoints for time
                    # that has already passed. Playing them would command
                    # the arms BACKWARD by however far they travelled in the
                    # meantime -- measured at 25-33mm per re-plan on real
                    # hardware. Drop them; _action_chunk_to_tape then takes
                    # num_open_loop_steps rows from what's left, so the tape
                    # length (and the prefetch trigger below) stay stable.
                else:
                    obs_dict = self._build_obs_dict(observation)
                    sent_state = obs_dict["observation/state"]
                    chunk = self._infer_chunk_blended(obs_dict)
                    stale_ticks = 0
                # Clamped so at least one row always survives: an empty tape
                # would read as is_fresh_chunk on the very next tick and spin
                # inference forever.
                drop = min(stale_ticks + self._action_latency_ticks, chunk.shape[0] - 1)
                tape = self._action_chunk_to_tape(chunk[drop:], observation)
                self._chunk_lengths[client_uuid] = len(tape)
                self._next_chunk_id[client_uuid] = self._next_chunk_id.get(client_uuid, -1) + 1
                self._chunk_ids[client_uuid] = self._next_chunk_id[client_uuid]
            elif len(tape) == self._prefetch_lead_ticks and client_uuid not in self._pending_infer:
                # Right after the FIRST pop of this chunk -- not the last,
                # which was this method's bug until 2026-09-01: triggering
                # at len(tape)==1 gives only ONE tick (~93-100ms) of real
                # head start, not the (num_open_loop_steps-1) the class
                # docstring always claimed. Harmless at samples_per_call=1
                # (~87ms/call fits in one tick by luck), but at
                # samples_per_call=4 (~350ms total) it turned into a real
                # ~255ms stall on every single re-plan -- confirmed on real
                # hardware: fresh-chunk tick dt jumped from ~0.09s to
                # ~0.35s the moment samples_per_call went from 1 to 4, and
                # 0.35s is almost exactly 4 * (a lone call's ~87ms).
                # Triggering here instead gives (num_open_loop_steps - 1)
                # ticks -- ~650ms at 8 steps/10 Hz -- comfortably covering
                # several draws, at the cost of the prefetch conditioning
                # on a marginally staler observation (1 tick old instead
                # of the freshest available). Per the sampling-variance
                # diagnosis this fix is built on, that trade is a good
                # one: pure per-call noise dominates the jitter far more
                # than one tick of staleness does -- PROVIDED the resulting
                # staleness is compensated for when the chunk is consumed,
                # which it now is (see the fresh-chunk branch above). It
                # wasn't, at first, and the uncompensated 7 ticks were much
                # worse than the stall they replaced.
                obs_dict = self._build_obs_dict(observation)
                self._pending_infer[client_uuid] = (
                    self._executor.submit(self._infer_chunk_blended, obs_dict),
                    obs_dict["observation/state"],
                    # Exactly one row is popped per tick, so however many
                    # rows remain now is how many ticks will elapse before
                    # this prefetch is consumed -- i.e. how stale it'll be.
                    len(tape),
                )
            step_within_chunk = self._chunk_lengths.get(client_uuid, 0) - len(tape)
            action = tape.pop(0)
            actions[client_uuid] = action
            self._action_tapes[client_uuid] = tape
            self._log_step(
                client_uuid=client_uuid,
                chunk_id=self._chunk_ids.get(client_uuid, -1),
                is_fresh_chunk=is_fresh_chunk,
                step_within_chunk=step_within_chunk,
                observation=observation,
                sent_state=sent_state,
                action=action,
            )
        return actions

    def _step_batch_ensembled(
        self, observations: dict[uuid.UUID, MultiarmObservation]
    ) -> dict[uuid.UUID, PosesAndGrippers]:
        actions: dict[uuid.UUID, PosesAndGrippers] = {}
        cadence = self._num_open_loop_steps
        for client_uuid, observation in observations.items():
            step = self._client_step.get(client_uuid, 0)
            live = self._live_chunks.setdefault(client_uuid, [])
            is_fresh_chunk = step % cadence == 0
            sent_state = None

            if is_fresh_chunk:
                pending = self._pending_infer.pop(client_uuid, None)
                if pending is not None:
                    future, sent_state, obs_step = pending
                    chunk = future.result()
                else:
                    obs_dict = self._build_obs_dict(observation)
                    sent_state = obs_dict["observation/state"]
                    chunk = self._infer_chunk_blended(obs_dict)
                    obs_step = step
                # Anchor the chunk to the tick whose OBSERVATION produced it,
                # not the tick it happens to arrive on. Row r of a chunk is
                # the absolute setpoint for obs_step + r, so a prefetched
                # chunk arriving `step - obs_step` ticks late must be entered
                # at that row, not at row 0. Anchoring at `step` instead --
                # what this did until 2026-09-01 -- replayed rows already in
                # the past and snapped both arms backward on every single
                # re-plan: measured over two rollouts, 86-93% of chunk
                # boundaries moved AGAINST the direction of travel, by
                # 22-28mm, undoing 0.6-0.9x of the previous 7 ticks of
                # motion. That is the "arms jerk back at every execution"
                # report, and it is a bookkeeping bug, not policy noise.
                live.append((obs_step, chunk))
                self._next_chunk_id[client_uuid] = self._next_chunk_id.get(client_uuid, -1) + 1
                self._chunk_ids[client_uuid] = self._next_chunk_id[client_uuid]

            # Drop chunks whose predicted horizon no longer covers this step.
            live[:] = [(start, c) for start, c in live if step - start + self._action_latency_ticks < c.shape[0]]
            if not live:
                raise RuntimeError(
                    f"no live chunk covers step {step} for client "
                    f"{client_uuid} -- reset() must run before the first "
                    "step(), and num_open_loop_steps must divide evenly "
                    "into however many steps have elapsed."
                )

            rows, weights, ages = [], [], []
            for start, chunk in live:
                age = step - start
                rows.append(chunk[age + self._action_latency_ticks])
                weights.append(np.exp(-self._ensemble_decay * age))
                ages.append(age)
            rows = np.stack(rows)
            blended = blend_actions20(rows, np.asarray(weights))
            if not self._ensemble_gripper:
                # Trust the freshest live chunk's gripper outright, same
                # reasoning as _infer_chunk_blended's per-draw vote: these
                # chunks disagree about grasp *phase*, not just noise, so
                # blending them can emit a command that never crosses the
                # close threshold.
                blended[GRIP_COLS] = rows[int(np.argmin(ages))][GRIP_COLS]
            action = self._decode_action_row(blended, observation)

            # Prefetch the chunk the NEXT cadence window will need,
            # prefetch_lead_ticks before the boundary -- enough to hide a
            # multi-draw _infer_chunk_blended (1 tick of lead stalled the
            # loop at 0.348s/tick with 4 draws), and no more than that,
            # since the arriving chunk is entered at row == lead and every
            # lead tick therefore costs a row of usable horizon. The
            # staleness itself is corrected for at the fresh-chunk branch
            # above via obs_step, so lead only trades responsiveness
            # against stalling -- it no longer causes a positional jump.
            if step % cadence == cadence - self._prefetch_lead_ticks and client_uuid not in self._pending_infer:
                obs_dict = self._build_obs_dict(observation)
                self._pending_infer[client_uuid] = (
                    self._executor.submit(self._infer_chunk_blended, obs_dict),
                    obs_dict["observation/state"],
                    step,
                )

            actions[client_uuid] = action
            self._client_step[client_uuid] = step + 1
            self._log_step(
                client_uuid=client_uuid,
                chunk_id=self._chunk_ids.get(client_uuid, -1),
                is_fresh_chunk=is_fresh_chunk,
                step_within_chunk=step % cadence,
                observation=observation,
                sent_state=sent_state,
                action=action,
            )
        return actions

    def reset_batch(self, seeds: dict[uuid.UUID, int | None], options=None) -> None:
        del options  # Unused: no policy-side state to seed.
        self._curate_known_clients(seeds.keys())
        for client_uuid in seeds:
            # anzu resets between demonstration indices, so a reset is an
            # episode boundary: close out whatever was being recorded before
            # any of the next episode's steps can land on it.
            if self._recorder is not None:
                self._recorder.end_episode(client_uuid)
            self._action_tapes[client_uuid] = []
            self._live_chunks[client_uuid] = []
            self._client_step[client_uuid] = 0
            self._step_counts[client_uuid] = 0
            pending = self._pending_infer.pop(client_uuid, None)
            if pending is not None:
                # Don't block reset() on an in-flight prefetch; just drop
                # it. Its result, if it lands late, is discarded (nothing
                # ever reads self._pending_infer[client_uuid] again once
                # popped, and reset already emptied the tape it would
                # have refilled).
                pending[0].cancel()

    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        return self.step_batch({self._internal_uuid: observation})[self._internal_uuid]

    def reset(self, *, seed=None, options=None) -> None:
        self.reset_batch({self._internal_uuid: seed}, options)

    def get_policy_metadata(self):
        return self._get_policy_metadata_impl("Pi0PolicyBatch")


@dc.dataclass
class Pi0PolicyWrapperConfig(PolicyConfig):
    """Mirrors `VLAPolicyWrapperConfig` / `DiffusionPolicyConfig`'s
    dataclass-with-`create()` shape. See `pi0_policy_server.py` for the CLI
    that fills this in.
    """

    websocket_host: str = "localhost"
    websocket_port: int = 8000
    api_key: str | None = None
    checkpoint_path: str = ""
    policy_config_name: str = "pi05_clean_spill"
    # Confirmed by tennyyyin on 2026-09-01 against the running davis station
    # (see the module docstring's ASSUMPTIONS note) -- these are the 3 of
    # davis's 6 cameras that feed this checkpoint's 3 image slots.
    base_camera: str = "scene_right_0"
    left_wrist_camera: str = "wrist_left_plus"
    right_wrist_camera: str = "wrist_right_plus"
    skill_type: str = "Undefined"
    num_open_loop_steps: int = 8
    client_timeout_s: float = DEFAULT_CLIENT_TIMEOUT_SECS
    batch: bool = True
    # If set, one JSON line per step() -- see Pi0PolicyWrapperBase._log_step.
    log_path: str | None = None
    # pi0.5 sampling-variance mitigation -- see action_blend.py and
    # Pi0PolicyWrapperBatch's docstring. Defaults match the proven values
    # from the sibling project this was ported from
    # (samples_per_call=4, ensemble_decay=0.3, ensemble_gripper=False);
    # samples_per_call=1 and ensemble_decay=None reproduce this bridge's
    # pre-fix behavior exactly.
    samples_per_call: int = 4
    ensemble_decay: float | None = 0.3
    ensemble_gripper: bool = False
    # Latency handling for a chunk of ABSOLUTE poses -- see the same-named
    # arguments on Pi0PolicyWrapperBase.__init__.
    action_latency_ticks: int = 0
    prefetch_lead_ticks: int = 5
    # Per-episode rollout recording -- see episode_recorder.py. `record_dir`
    # None disables it entirely; policy_name/task_shorthand default to the
    # checkpoint dir's basename and a snake_cased skill_type.
    record_dir: str | None = None
    task_shorthand: str | None = None
    policy_name: str | None = None
    record_fps: float = 10.0
    record_max_width: int = 640
    record_idle_timeout_s: float = 5.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        for field_name in ("base_camera", "left_wrist_camera", "right_wrist_camera"):
            value = getattr(self, field_name)
            assert value, f"`{field_name}` must not be empty."
            # Raises a clear KeyError if it's not a real camera name.
            SpartanCameraNames[value.upper()]
        assert self.num_open_loop_steps > 0
        assert self.client_timeout_s > 0.0
        assert self.samples_per_call >= 1
        assert self.ensemble_decay is None or self.ensemble_decay >= 0.0
        assert self.action_latency_ticks >= 0
        assert 1 <= self.prefetch_lead_ticks <= self.num_open_loop_steps
        assert self.record_fps > 0.0
        assert self.record_max_width >= 0
        assert self.record_idle_timeout_s > 0.0
        # Duplicated from Pi0PolicyWrapperBase.__init__ on purpose: that copy
        # only runs once a WebsocketClientPolicy has been constructed, so
        # checking here too means a bad CLI combination fails immediately
        # instead of after the connection attempt.
        deepest_row = self.prefetch_lead_ticks + self.num_open_loop_steps - 1 + self.action_latency_ticks
        assert deepest_row < _ACTION_HORIZON, (
            f"prefetch_lead_ticks({self.prefetch_lead_ticks}) + "
            f"num_open_loop_steps({self.num_open_loop_steps}) - 1 + "
            f"action_latency_ticks({self.action_latency_ticks}) = "
            f"{deepest_row} needs row {deepest_row} of a "
            f"{_ACTION_HORIZON}-row chunk."
        )

    def create(self):
        self.validate()
        kwargs = {
            "websocket_host": self.websocket_host,
            "websocket_port": self.websocket_port,
            "api_key": self.api_key,
            "checkpoint_path": self.checkpoint_path,
            "policy_config_name": self.policy_config_name,
            "base_camera": self.base_camera,
            "left_wrist_camera": self.left_wrist_camera,
            "right_wrist_camera": self.right_wrist_camera,
            "skill_type": self.skill_type,
            "num_open_loop_steps": self.num_open_loop_steps,
            "log_path": self.log_path,
            "samples_per_call": self.samples_per_call,
            "ensemble_decay": self.ensemble_decay,
            "ensemble_gripper": self.ensemble_gripper,
            "action_latency_ticks": self.action_latency_ticks,
            "prefetch_lead_ticks": self.prefetch_lead_ticks,
            "record_dir": self.record_dir,
            "task_shorthand": self.task_shorthand,
            "policy_name": self.policy_name,
            "record_fps": self.record_fps,
            "record_max_width": self.record_max_width,
            "record_idle_timeout_s": self.record_idle_timeout_s,
        }
        if self.batch:
            return Pi0PolicyWrapperBatch(**kwargs, client_timeout_s=self.client_timeout_s)
        return Pi0PolicyWrapper(**kwargs)
