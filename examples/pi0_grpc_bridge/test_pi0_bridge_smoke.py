"""Offline integration smoke test for pi0_policy_wrapper.py.

Exercises the FULL round trip -- observation construction, the real
openpi_client WebsocketClientPolicy, the real openpi
WebsocketPolicyServer wire protocol, and PosesAndGrippers decoding --
against a mock policy standing in for the real pi0.5 checkpoint. Needs no
GPU, no openpi server, no checkpoint: just this venv (see the run command
at the bottom).

What this DOES verify:
  * The 3 configured cameras + state + prompt get packed into the exact
    dict shape openpi expects.
  * A round trip through the real wire protocol (msgpack_numpy over
    websockets) works.
  * The [16, 20] action chunk decodes into 8 PosesAndGrippers (matching
    --num-open-loop-steps default) with sane arm/gripper names, and a
    9th step triggers exactly one more inference call (open-loop caching
    works).
  * Absolute (not relative) pose handling: a known input action decodes
    to the expected RigidTransform.

What this does NOT verify (needs the real checkpoint + serve_policy.py):
  * The state-vector joint ordering assumption (see module docstring in
    pi0_policy_wrapper.py).
  * Real image preprocessing/resizing behavior.
  * Actual policy outputs being sensible actions.
"""

import threading
import time

import numpy as np
from openpi_client.base_policy import BasePolicy
from pi0_policy_wrapper import _ACTION_HORIZON
from pi0_policy_wrapper import _STATE_DIM
from pi0_policy_wrapper import Pi0PolicyWrapperConfig
from pydrake.math import RigidTransform
from robot_gym.multiarm_spaces import CameraImageSet
from robot_gym.multiarm_spaces import CameraRgbImage
from robot_gym.multiarm_spaces import MultiarmObservation
from robot_gym.multiarm_spaces import PosesAndGrippers
from robot_gym.multiarm_spaces import PosesAndGrippersActualAndDesired

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class RecordingMockPolicy(BasePolicy):
    """Stands in for the real pi0.5 model. Records every obs dict it's
    given so the test can assert on exactly what the wrapper sent."""

    def __init__(self):
        self.received_obs = []
        self.infer_count = 0

    def infer(self, obs: dict) -> dict:
        self.infer_count += 1
        self.received_obs.append(obs)
        # A recognizable, non-zero action chunk so decoding is verifiable:
        # step t, dim d -> t * 100 + d (except we clamp gripper dims to
        # [0, 1]-ish ranges so PosesAndGrippers doesn't choke).
        chunk = np.zeros((_ACTION_HORIZON, 20), dtype=np.float32)
        for t in range(_ACTION_HORIZON):
            chunk[t, 0:3] = [0.1 * t, 0.2, 0.3]  # right xyz
            chunk[t, 3:9] = [1, 0, 0, 0, 1, 0]  # right rot6d = identity
            chunk[t, 9:12] = [-0.1 * t, -0.2, 0.4]  # left xyz
            chunk[t, 12:18] = [1, 0, 0, 0, 1, 0]  # left rot6d = identity
            chunk[t, 18] = 0.05  # right gripper
            chunk[t, 19] = 0.06  # left gripper
        return {"actions": chunk}


def _make_fake_observation() -> MultiarmObservation:
    identity = RigidTransform()
    actual = PosesAndGrippers(
        poses={"right::panda": identity, "left::panda": identity},
        grippers={"right::panda_hand": 0.1, "left::panda_hand": 0.1},
        joint_position={
            "right::panda": np.arange(7, dtype=np.float32),
            "left::panda": np.arange(7, 14, dtype=np.float32),
        },
    )
    robot = PosesAndGrippersActualAndDesired(actual=actual, desired=actual)

    def _cam(seed):
        img = np.ones((480, 640, 3), dtype=np.uint8) * seed
        return CameraImageSet(rgb=CameraRgbImage(array=img, K=np.eye(3), X_TC=identity))

    visuo = {
        "scene_right_0": _cam(10),
        "scene_left_0": _cam(20),
        "wrist_left_plus": _cam(30),
        "wrist_right_plus": _cam(40),
        "wrist_left_minus": _cam(50),
        "wrist_right_minus": _cam(60),
    }
    return MultiarmObservation(
        robot=robot,
        visuo=visuo,
        language_instruction="pick up the knocked over cup, set it upright, "
        "and wipe up the spilled liquid with a towel",
    )


def main():
    port = 8765
    mock_policy = RecordingMockPolicy()
    server = WebsocketPolicyServer(mock_policy, host="localhost", port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(1.0)  # let the server bind before the client connects

    config = Pi0PolicyWrapperConfig(
        websocket_host="localhost",
        websocket_port=port,
        checkpoint_path="/fake/for/testing",
        skill_type="BimanualCleanUpSpill",
        num_open_loop_steps=8,
        # Pinned off: this test is about tape/prefetch mechanics, not the
        # sampling-variance fix (covered separately by
        # test_samples_per_call_gripper_vote / test_ensemble_decay below) --
        # those defaulting to on would change every infer_count assertion.
        samples_per_call=1,
        ensemble_decay=None,
    )
    policy = config.create()
    print(f"Connected. Policy metadata: {policy.get_policy_metadata()}")

    obs = _make_fake_observation()

    # --- Assertion 1: exactly the 3 configured cameras are sent, right shapes.
    action = policy.step(obs)
    assert mock_policy.infer_count == 1, mock_policy.infer_count
    sent = mock_policy.received_obs[0]
    assert set(sent.keys()) == {
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
        "observation/state",
        "prompt",
    }, sent.keys()
    assert sent["observation/image"].mean() == 10, "base camera should be scene_right_0"
    assert sent["observation/left_wrist_image"].mean() == 30, "left wrist should be wrist_left_plus"
    assert sent["observation/right_wrist_image"].mean() == 40, "right wrist should be wrist_right_plus"
    assert sent["observation/state"].shape == (_STATE_DIM,)
    # left-first: joint_position["left::panda"] = arange(7,14), ["right::panda"] =
    # arange(7); both grippers 0.1. See pi0_policy_wrapper.py's module docstring --
    # this ordering is confirmed against convert_bike_rotor_to_lerobot.py, and
    # differs from the (right-first) action ordering asserted below.
    np.testing.assert_allclose(sent["observation/state"], [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 0.1, 0.1])
    assert sent["prompt"].startswith("pick up the knocked over cup")
    print("Assertion 1 OK: obs dict shape/contents/camera-mapping correct.")

    # --- Assertion 2: decoded action for step 0 matches the mock's known chunk.
    assert isinstance(action, PosesAndGrippers)
    right_pose = action.poses["right::panda"]
    np.testing.assert_allclose(right_pose.translation(), [0.0, 0.2, 0.3], atol=1e-5)
    assert action.grippers["right::panda_hand"] == np.float32(0.05)
    assert action.grippers["left::panda_hand"] == np.float32(0.06)
    print("Assertion 2 OK: absolute-pose action decoding correct for step 0.")

    # --- Assertion 3: open-loop caching -- the 8 steps of a chunk serve its
    # rows 0..7 in order, with no blocking re-infer.
    #
    # Asserted on the decoded rows, NOT on mock_policy.infer_count: the
    # prefetch fires mid-chunk in a background thread, so any count taken
    # here races the localhost round trip (an earlier version of this
    # assertion checked `infer_count == 1` and passed only by luck).
    xs = [action.poses["right::panda"].translation()[0]]
    xs.extend(policy.step(obs).poses["right::panda"].translation()[0] for _ in range(7))
    np.testing.assert_allclose(xs, [0.1 * t for t in range(8)], atol=1e-5)
    print("Assertion 3 OK: 8 open-loop steps served rows 0-7 of one chunk.")

    # --- Assertion 4: the 9th step consumes the PREFETCHED chunk, entered at
    # row prefetch_lead_ticks -- the latency compensation, and the whole
    # point of the 2026-09-01 fix.
    #
    # The prefetch was submitted 5 ticks before this one (default
    # prefetch_lead_ticks=5), so its rows 0..4 are setpoints for ticks that
    # have already happened. Entering at row 5 -> x == 0.5. Entering at row 0
    # (what this did before) -> x == 0.0, i.e. commanding the arm back to
    # where the chunk *started*, which on real hardware was a visible 25-33mm
    # backward jerk on every single re-plan, both arms.
    #
    # Only the ROW INDEX is checked, not seam continuity (x8 vs xs[-1]==0.7):
    # this mock returns an identical chunk whatever it's shown, so its "new
    # plan" always restarts from x=0 instead of continuing from where the arm
    # got to. A real policy conditioned on the tick-4 observation emits a
    # chunk whose row 0 is the setpoint for tick 4 -- which is what makes
    # entering at row 5 land on tick 9. Verifying that end-to-end needs the
    # real checkpoint on hardware (see analyze_trajectory.py).
    x8 = policy.step(obs).poses["right::panda"].translation()[0]
    assert mock_policy.infer_count == 2, mock_policy.infer_count
    np.testing.assert_allclose(x8, 0.5, atol=1e-5)
    print("Assertion 4 OK: prefetched chunk entered at row 5, not row 0.")

    # --- Assertion 5: reset() clears the tape and forces a fresh call on next step.
    policy.reset()
    policy.step(obs)
    assert mock_policy.infer_count == 3, mock_policy.infer_count
    print("Assertion 5 OK: reset() clears the action tape.")

    print("\nALL SMOKE TEST ASSERTIONS PASSED")


if __name__ == "__main__":
    main()


class VaryingMockPolicy(BasePolicy):
    """Returns a caller-controlled sequence of chunks, one per infer() call
    -- for testing samples_per_call/ensemble_decay, where consecutive
    calls must disagree the way independent stochastic draws would."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.infer_count = 0

    def infer(self, obs: dict) -> dict:
        chunk = self._chunks[self.infer_count % len(self._chunks)]
        self.infer_count += 1
        return {"actions": chunk.copy()}


def _flat_chunk(right_x, right_gripper, left_gripper):
    """[_ACTION_HORIZON, 20] chunk, identity rotations, constant per-row,
    only right-x and both grippers set -- enough to check blending without
    needing full pose realism."""
    chunk = np.zeros((_ACTION_HORIZON, 20), dtype=np.float32)
    chunk[:, 0] = right_x
    chunk[:, 3:9] = [1, 0, 0, 0, 1, 0]
    chunk[:, 12:18] = [1, 0, 0, 0, 1, 0]
    chunk[:, 18] = right_gripper
    chunk[:, 19] = left_gripper
    return chunk


def test_samples_per_call_gripper_vote():
    port = 8767
    # 3 draws: 2 say "closed" (~0.0), 1 says "open" (0.1) -- majority vote
    # must pick closed, NOT the mean (~0.033, which would be a phantom
    # half-open command). right_x varies too, to confirm pose still blends.
    mock = VaryingMockPolicy(
        [
            _flat_chunk(1.0, 0.0, 0.0),
            _flat_chunk(2.0, 0.1, 0.0),
            _flat_chunk(3.0, 0.0, 0.0),
        ]
    )
    server = WebsocketPolicyServer(mock, host="localhost", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(1.0)

    config = Pi0PolicyWrapperConfig(
        websocket_host="localhost",
        websocket_port=port,
        checkpoint_path="/fake",
        skill_type="Test",
        num_open_loop_steps=8,
        samples_per_call=3,
        ensemble_decay=None,
    )
    policy = config.create()
    obs = _make_fake_observation()
    action = policy.step(obs)
    assert mock.infer_count == 3, mock.infer_count  # exactly K draws, no more
    # pose: mean of 1,2,3 -> 2.0
    assert abs(action.poses["right::panda"].translation()[0] - 2.0) < 1e-4, action.poses["right::panda"].translation()[
        0
    ]
    # gripper: majority vote -> closed (0.0), NOT the mean (~0.033)
    assert action.grippers["right::panda_hand"] < 1e-6, action.grippers["right::panda_hand"]
    print("Assertion 6 OK: samples_per_call blends pose, majority-votes gripper.")


def test_ensemble_decay():
    port = 8768
    # Each fresh chunk (every num_open_loop_steps=2 ticks) reports its own
    # "generation" as right_x so we can see which chunks are contributing.
    mock = VaryingMockPolicy(
        [
            _flat_chunk(10.0, 0.1, 0.1),  # chunk 0, all rows x=10
            _flat_chunk(20.0, 0.1, 0.1),  # chunk 1, all rows x=20
            _flat_chunk(30.0, 0.1, 0.1),  # chunk 2
        ]
    )
    server = WebsocketPolicyServer(mock, host="localhost", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(1.0)

    config = Pi0PolicyWrapperConfig(
        websocket_host="localhost",
        websocket_port=port,
        checkpoint_path="/fake",
        skill_type="Test",
        num_open_loop_steps=2,
        samples_per_call=1,
        ensemble_decay=0.5,
        ensemble_gripper=False,
        # Must be <= num_open_loop_steps; at cadence 2 the default 5 is
        # rejected outright by validate().
        prefetch_lead_ticks=1,
    )
    policy = config.create()
    obs = _make_fake_observation()

    x0 = policy.step(obs).poses["right::panda"].translation()[0]
    assert mock.infer_count == 1
    assert abs(x0 - 10.0) < 1e-4, x0  # only chunk 0 alive, age 0 -> exactly 10

    x1 = policy.step(obs).poses["right::panda"].translation()[0]
    assert mock.infer_count == 1  # still tick 1 of cadence=2, no new call
    assert abs(x1 - 10.0) < 1e-4, x1  # chunk 0, age 1 -> still just 10

    x2 = policy.step(obs).poses["right::panda"].translation()[0]
    assert mock.infer_count == 2  # tick 2 == new cadence window -> re-infer
    # Now TWO live chunks cover this step: chunk0 (age=2) and chunk1 (age=0).
    # Weighted blend must land strictly between 10 and 20, closer to 20
    # (chunk1 is fresher, so it gets more weight at decay=0.5) --
    # not exactly 20, which is what "always trust newest" would give.
    assert 10.0 < x2 < 20.0, x2
    assert x2 > 15.0, f"expected fresher chunk to dominate, got {x2}"
    print(
        f"Assertion 7 OK: ensemble blends across live chunks (x2={x2:.3f}, "
        f"between the two live chunks' values, fresher-weighted)."
    )

    # Gripper must come from the freshest chunk only, never blended --
    # both mock chunks command 0.1 here so this mainly checks no crash /
    # no NaN from the GRIP_COLS override path.
    g2 = policy.step(obs).grippers["right::panda_hand"]
    assert abs(g2 - 0.1) < 1e-4, g2
    print("Assertion 8 OK: ensemble gripper handling doesn't blend/crash.")


def _ramp_chunk():
    """A chunk whose right-x IS its row index, so a decoded action reveals
    exactly which row of the chunk was used."""
    chunk = np.zeros((_ACTION_HORIZON, 20), dtype=np.float32)
    chunk[:, 0] = np.arange(_ACTION_HORIZON, dtype=np.float32)
    chunk[:, 3:9] = [1, 0, 0, 0, 1, 0]
    chunk[:, 12:18] = [1, 0, 0, 0, 1, 0]
    return chunk


def test_ensemble_anchors_prefetch_to_its_observation_step():
    """A prefetched chunk must be entered at the row matching how stale it
    is, on the ENSEMBLE path -- the one that actually runs on hardware.

    Regression test for the 2026-09-01 backward-jerk bug: chunks were
    appended to the live list as `(arrival_step, chunk)` instead of
    `(observation_step, chunk)`, so a chunk inferred N ticks earlier was
    replayed from row 0 and re-commanded N ticks of already-executed
    motion. Measured on two rollouts: 86-93% of chunk boundaries moved
    against the direction of travel by 22-28mm, both arms, every re-plan.
    """
    port = 8769
    mock = VaryingMockPolicy([_ramp_chunk(), _ramp_chunk(), _ramp_chunk()])
    server = WebsocketPolicyServer(mock, host="localhost", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(1.0)

    cadence, lead = 4, 2
    config = Pi0PolicyWrapperConfig(
        websocket_host="localhost",
        websocket_port=port,
        checkpoint_path="/fake",
        skill_type="Test",
        num_open_loop_steps=cadence,
        samples_per_call=1,
        # Steep decay so the freshest chunk carries ~all the weight and the
        # blend reports (near enough) the row IT contributed -- this test is
        # about row selection, not about the blend itself.
        ensemble_decay=5.0,
        prefetch_lead_ticks=lead,
    )
    policy = config.create()
    obs = _make_fake_observation()

    xs = [policy.step(obs).poses["right::panda"].translation()[0] for _ in range(2 * cadence)]

    # Chunk 0 is synchronous (obs_step == 0), so rows 0..3 -> x 0,1,2,3.
    np.testing.assert_allclose(xs[:cadence], [0, 1, 2, 3], atol=1e-4)
    # Chunk 1 was prefetched at step cadence-lead == 2, so at step 4 it is
    # `lead` == 2 ticks stale and must be entered at row 2, giving 2,3,4,5 --
    # i.e. row 2 onward rather than restarting at row 0.
    # (Row index only -- see Assertion 4 for why an observation-independent
    # mock can't demonstrate seam continuity itself.)
    np.testing.assert_allclose(xs[cadence:], [2, 3, 4, 5], atol=1e-2)
    print(
        f"Assertion 9 OK: ensemble enters a {lead}-tick-stale prefetched "
        f"chunk at row {lead} (xs={[round(x, 2) for x in xs]})."
    )


def test_horizon_budget_is_validated_up_front():
    """lead + cadence - 1 + latency must fit inside the action horizon, and
    must be rejected at construction rather than as a mid-rollout
    RuntimeError from _step_batch_ensembled."""
    for kwargs in (
        # 8 + 8 - 1 = 15 rows of lead+cadence is fine, +2 latency is not.
        {"num_open_loop_steps": 8, "prefetch_lead_ticks": 8, "action_latency_ticks": 2},
        # Lead cannot exceed the cadence.
        {"num_open_loop_steps": 4, "prefetch_lead_ticks": 6},
    ):
        try:
            Pi0PolicyWrapperConfig(checkpoint_path="/fake", skill_type="Test", **kwargs)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"should have been rejected: {kwargs}")
    print("Assertion 10 OK: impossible horizon budgets rejected at config time.")


def test_episode_recording():
    """End-to-end per-episode recording (episode_recorder.py) through the
    real wrapper: 3 videos + commanded/reached state trajectory, saved when
    the episode ends.

    Also checks the videos aren't view-swapped, which is the reason this
    exists: `_make_fake_observation` gives each camera a distinct constant
    brightness (base=10, left wrist=30, right wrist=40), so a swapped
    --left-wrist-camera/--right-wrist-camera shows up as the wrong mean in
    the wrong file. (On hardware the equivalent check is: does
    left_wrist_*.mp4 show the LEFT gripper.)
    """
    import csv
    import json
    import os
    import shutil
    import tempfile

    from episode_recorder import read_video_frames

    port = 8770
    mock = VaryingMockPolicy([_ramp_chunk() for _ in range(8)])
    server = WebsocketPolicyServer(mock, host="localhost", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(1.0)

    root = tempfile.mkdtemp(prefix="pi0-record-smoke-")
    try:
        config = Pi0PolicyWrapperConfig(
            websocket_host="localhost",
            websocket_port=port,
            checkpoint_path="/fake/ckpts/r2_1750",
            skill_type="BimanualCleanUpSpill",
            num_open_loop_steps=2,
            prefetch_lead_ticks=1,
            samples_per_call=1,
            ensemble_decay=0.1,
            record_dir=root,
            # Long enough that only the explicit reset below ends the
            # episode -- the idle path is covered by episode_recorder.py's
            # own self-test.
            record_idle_timeout_s=600.0,
        )
        policy = config.create()
        obs = _make_fake_observation()

        policy.reset()
        n_steps = 6
        for _ in range(n_steps):
            policy.step(obs)
        # anzu resets between demonstration indices; that closes the episode.
        policy.reset()

        # Folder name: <checkpoint basename>_<snake_cased skill>.
        run_root = os.path.join(root, "r2_1750_clean_up_spill")
        assert os.path.isdir(run_root), sorted(os.listdir(root))
        episodes = sorted(os.listdir(run_root))
        assert len(episodes) == 1, episodes
        episode_dir = os.path.join(run_root, episodes[0])

        with open(os.path.join(episode_dir, "meta.json")) as f:
            meta = json.load(f)
        assert meta["n_steps"] == n_steps, meta
        assert meta["n_dropped_steps"] == 0, meta
        assert not meta["recording_failed"], meta
        assert meta["policy_name"] == "r2_1750", meta
        assert meta["language_instruction"].startswith("pick up the knocked"), meta

        with open(os.path.join(episode_dir, "states.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n_steps, len(rows)
        # Commanded and reached are both populated, and distinguishable: the
        # mock's ramp chunk commands right-x = the row index while the fake
        # observation's measured pose sits at the identity.
        assert float(rows[0]["meas_right_x"]) == 0.0, rows[0]
        assert float(rows[-1]["cmd_right_x"]) > 0.0, rows[-1]
        # Reached joints come through (7 per arm, from joint_position).
        assert float(rows[0]["meas_left_joint_6"]) == 13.0, rows[0]
        assert float(rows[0]["meas_right_joint_6"]) == 6.0, rows[0]
        # The 16-d policy input is recorded on inference ticks and blank on
        # the others -- never a stale copy. cadence=2, so every other tick.
        assert rows[0]["state_0"] != "", rows[0]
        assert rows[1]["state_0"] == "", rows[1]

        expected_mean = {
            "base_scene_right_0.mp4": 10,
            "left_wrist_wrist_left_plus.mp4": 30,
            "right_wrist_wrist_right_plus.mp4": 40,
        }
        assert sorted(os.listdir(episode_dir)) == sorted([*expected_mean, "meta.json", "states.csv", "states.jsonl"]), (
            sorted(os.listdir(episode_dir))
        )
        for name, mean in expected_mean.items():
            frames = read_video_frames(os.path.join(episode_dir, name))
            assert len(frames) == n_steps, (name, len(frames))
            assert frames[0].shape[:2] == (480, 640), (name, frames[0].shape)
            # h264 is lossy, so this is a tolerance not an equality; the
            # three expected values are 10 apart, far outside it.
            got = float(np.mean(frames[0]))
            assert abs(got - mean) < 3.0, (name, got, mean)
        print(
            f"Assertion 11 OK: episode recorded to {episodes[0]} -- "
            f"3 unswapped views x {n_steps} frames, {len(rows)} state rows."
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_samples_per_call_gripper_vote()
    test_ensemble_decay()
    test_ensemble_anchors_prefetch_to_its_observation_step()
    test_horizon_budget_is_validated_up_front()
    test_episode_recording()
    print("\nEXTENDED (samples_per_call / ensemble_decay / latency / recording) ASSERTIONS PASSED")
