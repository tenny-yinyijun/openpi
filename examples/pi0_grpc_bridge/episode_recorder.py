"""Per-episode rollout recording for the pi0.5 gRPC bridge.

Writes one directory per rollout episode:

    <root>/<policy-name>_<task-shorthand>/<YYYYmmddTHHMMSS>/
        base_scene_right_0.mp4          <- exactly the frames the policy saw
        left_wrist_wrist_left_plus.mp4
        right_wrist_wrist_right_plus.mp4
        states.csv                      <- commanded + reached, one row per tick
        states.jsonl                    <- same data, unflattened
        meta.json                       <- policy/config/camera provenance

Why this lives in the bridge rather than in anzu: the bridge already
receives the full `MultiarmObservation` (every camera view plus measured
poses/joints) on EVERY control tick and is the thing that computes the
commanded action, so commanded and reached are recorded from the same
object at the same instant -- no clock or index alignment to get wrong. It
also already knows episode boundaries, because anzu calls `reset_batch`
between demonstrations.

The videos are the 3 views *as fed to the model* (same
`--base-camera`/`--left-wrist-camera`/`--right-wrist-camera` lookup, and
the camera name is in the filename), so a swapped or dead wrist camera is
visible directly -- that is what this was built to answer.

Three hard rules, all because a robot is moving while this runs:

1. Never block the ~10 Hz control loop. Encoding happens on a background
   thread fed by a bounded queue; if the encoder ever falls behind, the
   step is DROPPED (both its frames and its state row, together, so video
   and CSV never drift out of sync) and counted in meta.json.
2. Never crash a rollout. Every recording failure is caught, reported
   once, and turns recording off for that episode. Losing data is bad;
   dropping a robot mid-motion is worse.
3. NEVER FORK. Nothing here may spawn a subprocess, because this code runs
   inside the gRPC server process while anzu holds a live connection to it.
   The first version of this file encoded via `imageio` + `libx264`, which
   shells out to the ffmpeg CLI: opening the 3 writers on the first step of
   an episode forked 3 times, gRPC logged

       fork_posix.cc:71] Other threads are currently calling into gRPC,
       skipping fork() handlers

   (its atfork handlers cannot quiesce a transport that worker threads are
   mid-call on), the server's HTTP/2 connection state was left corrupt, and
   the very next observation killed the whole 100-episode run with

       StatusCode.UNAVAILABLE  "Failed parsing HTTP/2 (Frame size 6911837
       is larger than max frame size 4194304)"

   -- the "frame size" being the desynchronised parser reading the gRPC
   message length prefix as a frame header, not any limit worth raising.
   Reproduced exactly once on 2026-09-01, on the first step of the first
   episode after recording was added. Hence PyAV: it links libavcodec into
   this process, so the same H.264 output costs no subprocess. If you ever
   swap the encoder, verify with `pgrep -P <server-pid>` that a recorded
   episode spawns no children.
"""

import contextlib
import csv
from fractions import Fraction
import json
import os
import queue
import threading
import time
import traceback
import typing

import av
import numpy as np

# Only used to downscale frames. Imported here rather than lazily in the
# writer thread on the first frame: this module is imported at server
# startup, so a slow or broken cv2 shows up then instead of mid-rollout.
try:
    import cv2
except Exception:  # pragma: no cover -- exercised only on a broken install
    cv2 = None

# Written by the writer thread only; the control-loop thread just enqueues.
_SENTINEL = object()

# ~12 s of backlog at 10 Hz. Bounded so a stalled encoder can't grow into
# the RAM the policy server needs (3 views x ~1 MB/frame adds up fast).
_QUEUE_MAXSIZE = 120


def _log(msg: str) -> None:
    """`print` with an explicit flush.

    These `[record]` lines are the only visibility into recording, and the
    bridge is normally started with its stdout redirected to a log file --
    where Python block-buffers instead of line-buffering, so without the
    flush a whole run's worth of them sits in the buffer and never reaches
    the file if the server is killed rather than exiting cleanly. Which is
    exactly how it's stopped.
    """
    print(msg, flush=True)


def default_task_shorthand(skill_type: str) -> str:
    """`BimanualCleanUpSpill` -> `clean_up_spill`.

    Drops a leading `Bimanual` (it's true of every skill on this station,
    so it carries no information) and converts CamelCase to snake_case.
    Overridable via --task-shorthand for anything this gets wrong.
    """
    name = skill_type
    for prefix in ("Bimanual", "Unimanual"):
        name = name.removeprefix(prefix)
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out) or skill_type.lower()


def default_policy_name(checkpoint_path: str) -> str:
    """Last path component of the checkpoint dir -- `.../ckpts/base` ->
    `base`, `.../r2_1750` -> `r2_1750`. Works for s3:// URIs too. Falls
    back to `unknown_policy` if --checkpoint-path wasn't passed (it's
    metadata-only for the server itself, so it can legitimately be empty).
    """
    stripped = checkpoint_path.rstrip("/")
    return os.path.basename(stripped) if stripped else "unknown_policy"


def _to_even_uint8_rgb(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Make `frame` safe for an h264 yuv420p stream: RGB uint8, contiguous,
    downscaled to at most `max_width`, with even height/width (yuv420p
    subsamples 2x2, so odd dimensions are rejected outright).
    """
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3 RGB frame, got shape {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        # Both conventions show up across drivers: float images are
        # normally 0-1, ints already 0-255.
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255)
        arr = arr.astype(np.uint8)

    h, w = arr.shape[:2]
    if max_width and w > max_width:
        scale = max_width / float(w)
        new_w, new_h = round(w * scale), round(h * scale)
        if cv2 is not None:
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Nearest-neighbour decimation: worse looking, but this is a
            # debugging artifact and a missing cv2 must not cost the
            # recording entirely.
            arr = arr[:: max(1, h // new_h), :: max(1, w // new_w)]
        h, w = arr.shape[:2]
    if h % 2 or w % 2:
        arr = arr[: h - (h % 2), : w - (w % 2)]
    return np.ascontiguousarray(arr)


# Column layout for states.csv. `cmd_*` is what the bridge commanded this
# tick (blended/decoded absolute pose); `meas_*` is what the station
# reported measured at the same tick, including joint angles -- "commanded
# and reached", as requested. `state_*` is the exact 16-d vector handed to
# the model, present only on ticks that ran inference (empty otherwise:
# non-inference ticks build no observation, and writing a stale copy would
# make the log claim something that never went over the wire).
_ARMS = ("right", "left")
_POSE_SUFFIXES = ("x", "y", "z", "qw", "qx", "qy", "qz")


def _child_pids() -> set[str]:
    """Every child process of this process, across all its threads.

    Used by `_self_test` to enforce rule 3 (never fork). Reads
    /proc/<pid>/task/<tid>/children, which is per-thread -- the writer
    thread is where an encoder would spawn one.
    """
    pids: set[str] = set()
    for tid in os.listdir("/proc/self/task"):
        # A thread can exit between listdir and open, hence the suppress.
        with contextlib.suppress(OSError), open(f"/proc/self/task/{tid}/children") as f:
            pids.update(f.read().split())
    return pids


def read_video_frames(path: str) -> list[np.ndarray]:
    """Decode an episode video back to a list of RGB frames.

    Lives here rather than in the tests because both this module's
    `_self_test` and `test_pi0_bridge_smoke.py` need it: reading back what
    was written is the only way to prove the three views aren't swapped.
    """
    with av.open(path) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


class _H264Writer:
    """One H.264/mp4 stream, encoded IN THIS PROCESS via PyAV.

    The no-subprocess property is the whole point -- see rule 3 in the
    module docstring. Frames are sized from the first one appended rather
    than up front, so a view whose camera only starts producing mid-episode
    still records instead of erroring.
    """

    def __init__(self, path: str, fps: float):
        self._container = av.open(path, mode="w")
        # PyAV wants an exact rational rate; --record-fps is a float.
        self._stream = self._container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        self._stream.pix_fmt = "yuv420p"
        # ultrafast keeps the writer thread far ahead of a 10 Hz feed even
        # with 3 streams; crf 23 is visually fine for inspection and keeps
        # an episode to a few MB.
        self._stream.options = {"preset": "ultrafast", "crf": "23"}
        self._sized = False

    def append(self, frame: np.ndarray) -> None:
        if not self._sized:
            self._stream.height, self._stream.width = frame.shape[:2]
            self._sized = True
        for packet in self._stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            self._container.mux(packet)

    def close(self) -> None:
        try:
            if self._sized:
                # Flush the encoder's lookahead; without this the last few
                # frames never reach the file.
                for packet in self._stream.encode():
                    self._container.mux(packet)
        finally:
            self._container.close()


def _csv_header(state_dim: int) -> list[str]:
    cols = ["t", "rel_t", "step", "chunk_id", "step_within_chunk", "is_fresh_chunk"]
    for kind in ("cmd", "meas"):
        for arm in _ARMS:
            cols += [f"{kind}_{arm}_{s}" for s in _POSE_SUFFIXES]
        for arm in _ARMS:
            cols.append(f"{kind}_{arm}_gripper")
    for arm in _ARMS:
        cols += [f"meas_{arm}_joint_{i}" for i in range(7)]
    cols += [f"state_{i}" for i in range(state_dim)]
    return cols


def _csv_row(record: dict, t0: float, state_dim: int) -> list:
    row = [
        f"{record['t']:.6f}",
        f"{record['t'] - t0:.6f}",
        record.get("step", ""),
        record.get("chunk_id", ""),
        record.get("step_within_chunk", ""),
        int(bool(record.get("is_fresh_chunk"))),
    ]
    for key in ("commanded", "measured"):
        block = record[key]
        for arm in _ARMS:
            row += [f"{v:.6f}" for v in block[f"{arm}_pos"]]
            row += [f"{v:.6f}" for v in block.get(f"{arm}_quat", [""] * 4)]
        for arm in _ARMS:
            row.append(f"{block[f'{arm}_gripper']:.6f}")
    for arm in _ARMS:
        joints = record["measured"].get(f"{arm}_joints") or [""] * 7
        row += [f"{v:.6f}" if v != "" else "" for v in joints]
    state = record.get("sent_state")
    row += [f"{v:.6f}" for v in state] if state is not None else [""] * state_dim
    return row


class EpisodeRecorder:
    """One episode. Thread-safe for a single producer (the control loop)
    plus this object's own writer thread.

    Nothing is created on disk until the first step arrives, so a client
    that connects and resets without ever stepping leaves no empty
    directory behind.
    """

    def __init__(
        self,
        *,
        episode_dir: str,
        view_cameras: dict[str, str],
        meta: dict,
        fps: float,
        max_width: int,
        state_dim: int,
    ):
        self._dir = episode_dir
        self._view_cameras = dict(view_cameras)
        self._meta = dict(meta)
        self._fps = fps
        self._max_width = max_width
        self._state_dim = state_dim

        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._writers: dict[str, typing.Any] = {}
        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None
        self._t0 = None
        self._n_written = 0
        self._n_dropped = 0
        self._started_wall = None
        self._failed = False
        self._closed = False
        self.last_step_time = time.time()

        self._thread = threading.Thread(target=self._run, name="pi0-record", daemon=True)
        self._thread.start()

    # ---- control-loop side (must never block or raise) ----

    def add_step(self, record: dict, frames: dict[str, np.ndarray]) -> None:
        self.last_step_time = time.time()
        if self._failed or self._closed:
            return
        try:
            # Frames are copied because the station's driver may reuse its
            # buffers; the writer thread reads them much later.
            payload = (
                record,
                {name: np.array(arr, copy=True) for name, arr in frames.items()},
            )
            self._queue.put_nowait(payload)
        except queue.Full:
            self._n_dropped += 1
        except Exception:
            self._n_dropped += 1

    def close(self, *, join_timeout_s: float = 30.0) -> str | None:
        """Flush and finalize. Returns the episode dir if anything was
        written, else None. Safe to call twice."""
        if self._closed:
            return self._dir if self._n_written else None
        self._closed = True
        with contextlib.suppress(queue.Full):
            self._queue.put(_SENTINEL, timeout=5.0)
        self._thread.join(timeout=join_timeout_s)
        if self._thread.is_alive():
            _log(f"[record] WARNING writer thread for {self._dir} did not finish in time; video may be truncated")
        return self._dir if self._n_written else None

    # ---- writer-thread side ----

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                if self._failed:
                    continue
                try:
                    self._write_step(*item)
                except Exception:
                    self._failed = True
                    _log(
                        f"[record] recording DISABLED for {self._dir} after "
                        "an error (the rollout itself is unaffected):\n" + traceback.format_exc()
                    )
        finally:
            self._finalize()

    def _write_step(self, record: dict, frames: dict[str, np.ndarray]):
        if self._t0 is None:
            self._open(record)
        for view, writer in self._writers.items():
            frame = frames.get(view)
            if frame is None:
                continue
            writer.append(_to_even_uint8_rgb(frame, self._max_width))
        self._csv_writer.writerow(_csv_row(record, self._t0, self._state_dim))
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._n_written += 1

    def _open(self, record: dict) -> None:
        os.makedirs(self._dir, exist_ok=True)
        self._t0 = record["t"]
        self._started_wall = time.time()
        for view, camera in self._view_cameras.items():
            path = os.path.join(self._dir, f"{view}_{camera}.mp4")
            self._writers[view] = _H264Writer(path, self._fps)
        # Both handles outlive _open() by design: one per episode,
        # appended to on every tick and closed in _finalize() -- hence the
        # SIM115 waivers rather than a context manager.
        self._csv_file = open(  # noqa: SIM115
            os.path.join(self._dir, "states.csv"), "w", newline=""
        )
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(_csv_header(self._state_dim))
        self._jsonl_file = open(  # noqa: SIM115
            os.path.join(self._dir, "states.jsonl"), "w"
        )
        _log(f"[record] recording episode to {self._dir}")

    def _finalize(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                _log("[record] WARNING failed to close a video writer:\n" + traceback.format_exc())
        for handle in (self._csv_file, self._jsonl_file):
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
        if self._t0 is None:
            return  # Nothing ever arrived; nothing on disk to describe.
        meta = dict(self._meta)
        meta.update(
            views=dict(self._view_cameras),
            video_files={v: f"{v}_{c}.mp4" for v, c in self._view_cameras.items()},
            fps=self._fps,
            n_steps=self._n_written,
            n_dropped_steps=self._n_dropped,
            recording_failed=self._failed,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._started_wall)),
            duration_s=round(time.time() - self._started_wall, 3),
        )
        try:
            with open(os.path.join(self._dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True, default=str)
        except Exception:
            _log("[record] WARNING failed to write meta.json:\n" + traceback.format_exc())
        note = f" ({self._n_dropped} steps dropped)" if self._n_dropped else ""
        _log(f"[record] saved {self._n_written} steps to {self._dir}{note}")


class RolloutRecorder:
    """Owns one `EpisodeRecorder` per anzu client, and decides when an
    episode is over.

    Episodes end on whichever comes first:

    * `end_episode()` -- anzu called `reset_batch`, i.e. it's setting up
      for the next demonstration index.
    * idle timeout -- no step for `idle_timeout_s`. This is the one that
      matters in practice: after the operator marks success/failure the
      rollout stops stepping and anzu blocks on its rubric GUI, which can
      be a long time before the next reset. The watchdog means the data is
      on disk seconds after the run stops, not whenever anzu gets around
      to the next episode.
    * `close()` -- server shutting down.
    """

    def __init__(
        self,
        *,
        root_dir: str,
        policy_name: str,
        task_shorthand: str,
        state_dim: int,
        fps: float = 10.0,
        max_width: int = 640,
        idle_timeout_s: float = 5.0,
        base_meta: dict | None = None,
    ):
        self._root = os.path.join(root_dir, f"{policy_name}_{task_shorthand}")
        self._policy_name = policy_name
        self._task_shorthand = task_shorthand
        self._state_dim = state_dim
        self._fps = fps
        self._max_width = max_width
        self._idle_timeout_s = idle_timeout_s
        self._base_meta = dict(base_meta or {})
        self._episodes: dict[typing.Any, EpisodeRecorder] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._used_dirs: set[str] = set()
        _log(f"[record] episodes will be written under {self._root}")
        self._watchdog = threading.Thread(target=self._watch_idle, name="pi0-record-idle", daemon=True)
        self._watchdog.start()

    def _new_dir(self) -> str:
        """Timestamped, and guaranteed not to collide: two clients (or a
        fast reset) can start inside the same second, and silently
        appending both to one directory would interleave two episodes."""
        base = time.strftime("%Y%m%dT%H%M%S")
        candidate = os.path.join(self._root, base)
        suffix = 1
        while candidate in self._used_dirs or os.path.exists(candidate):
            candidate = os.path.join(self._root, f"{base}_{suffix}")
            suffix += 1
        self._used_dirs.add(candidate)
        return candidate

    def record_step(
        self,
        client_uuid,
        record: dict,
        frames: dict[str, np.ndarray],
        view_cameras: dict[str, str],
        extra_meta: dict | None = None,
    ) -> None:
        if self._stopping.is_set():
            return
        with self._lock:
            episode = self._episodes.get(client_uuid)
            if episode is None:
                meta = dict(self._base_meta)
                meta.update(extra_meta or {})
                meta.update(
                    policy_name=self._policy_name,
                    task=self._task_shorthand,
                    client=str(client_uuid),
                )
                episode = EpisodeRecorder(
                    episode_dir=self._new_dir(),
                    view_cameras=view_cameras,
                    meta=meta,
                    fps=self._fps,
                    max_width=self._max_width,
                    state_dim=self._state_dim,
                )
                self._episodes[client_uuid] = episode
        episode.add_step(record, frames)

    def end_episode(self, client_uuid) -> None:
        with self._lock:
            episode = self._episodes.pop(client_uuid, None)
        if episode is not None:
            # Outside the lock: close() joins the writer thread, which can
            # take a moment to drain, and holding the lock would stall any
            # other client's step.
            episode.close()

    def close(self) -> None:
        self._stopping.set()
        with self._lock:
            episodes = list(self._episodes.values())
            self._episodes.clear()
        for episode in episodes:
            episode.close()

    def _watch_idle(self) -> None:
        while not self._stopping.wait(1.0):
            now = time.time()
            with self._lock:
                stale = [
                    client for client, ep in self._episodes.items() if now - ep.last_step_time > self._idle_timeout_s
                ]
                episodes = [self._episodes.pop(c) for c in stale]
            for episode in episodes:
                _log(f"[record] no steps for {self._idle_timeout_s:.0f}s -- closing episode")
                episode.close()


def _self_test():
    """`python episode_recorder.py` -- exercises the whole writer path
    (encoder, CSV, meta, idle close) on synthetic data, no robot needed."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="pi0-record-selftest-")
    try:
        rec = RolloutRecorder(
            root_dir=root,
            policy_name="base",
            task_shorthand=default_task_shorthand("BimanualCleanUpSpill"),
            state_dim=16,
            fps=10.0,
            idle_timeout_s=1.0,
            base_meta={"checkpoint_path": "/fake/ckpts/base"},
        )
        views = {
            "base": "scene_right_0",
            "left_wrist": "wrist_left_plus",
            "right_wrist": "wrist_right_plus",
        }
        for i in range(20):
            record = {
                "t": 1000.0 + 0.1 * i,
                "step": i,
                "chunk_id": i // 2,
                "step_within_chunk": i % 2,
                "is_fresh_chunk": i % 2 == 0,
                "measured": {
                    "right_pos": [0.001 * i, 0.2, 0.3],
                    "left_pos": [-0.001 * i, -0.2, 0.4],
                    "right_quat": [1.0, 0.0, 0.0, 0.0],
                    "left_quat": [1.0, 0.0, 0.0, 0.0],
                    "right_joints": list(np.arange(7) * 0.01),
                    "left_joints": list(np.arange(7) * 0.02),
                    "right_gripper": 0.05,
                    "left_gripper": 0.06,
                },
                "commanded": {
                    "right_pos": [0.001 * i + 0.01, 0.2, 0.3],
                    "left_pos": [-0.001 * i - 0.01, -0.2, 0.4],
                    "right_quat": [1.0, 0.0, 0.0, 0.0],
                    "left_quat": [1.0, 0.0, 0.0, 0.0],
                    "right_gripper": 0.05,
                    "left_gripper": 0.06,
                },
                "sent_state": list(np.arange(16, dtype=float)) if i % 2 == 0 else None,
            }
            # Odd 641x481 on purpose: exercises the even-dimension fix.
            frames = {v: np.random.randint(0, 255, (481, 641, 3), dtype=np.uint8) for v in views}
            rec.record_step("client-a", record, frames, views)
        # Rule 3: encoding 20 frames x 3 views must not have forked. This is
        # the assertion that would have caught the imageio/ffmpeg-CLI bug
        # that killed a live rollout on 2026-09-01.
        assert not _child_pids(), f"recording spawned subprocesses: {sorted(_child_pids())}"
        # Don't call end_episode: prove the idle watchdog closes it, which
        # is the path that actually fires after success/failure.
        time.sleep(3.0)
        episode_dirs = [
            os.path.join(root, "base_clean_up_spill", d)
            for d in sorted(os.listdir(os.path.join(root, "base_clean_up_spill")))
        ]
        assert len(episode_dirs) == 1, episode_dirs
        files = sorted(os.listdir(episode_dirs[0]))
        assert files == [
            "base_scene_right_0.mp4",
            "left_wrist_wrist_left_plus.mp4",
            "meta.json",
            "right_wrist_wrist_right_plus.mp4",
            "states.csv",
            "states.jsonl",
        ], files
        with open(os.path.join(episode_dirs[0], "states.csv")) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 21, len(lines)  # header + 20 steps
        with open(os.path.join(episode_dirs[0], "meta.json")) as f:
            meta = json.load(f)
        assert meta["n_steps"] == 20, meta
        assert meta["n_dropped_steps"] == 0, meta
        assert not meta["recording_failed"], meta
        for name in files:
            if not name.endswith(".mp4"):
                continue
            frames_read = read_video_frames(os.path.join(episode_dirs[0], name))
            assert len(frames_read) == 20, (name, len(frames_read))
            assert frames_read[0].shape[:2] == (480, 640), frames_read[0].shape
        rec.close()
        print("\nEPISODE RECORDER SELF-TEST PASSED (3 videos x 20 frames, 20 CSV rows, idle-close, meta.json)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    _self_test()
