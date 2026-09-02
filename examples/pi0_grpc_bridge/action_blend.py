"""SO(3)-aware blending for 20-d ``xyzrot6g`` bimanual actions.

Ported from `openworld/autoregressive/conditioning/action_adapter.py`
(`blend_actions20`, `rot6d_to_matrix`, `matrix_to_rot6d`) on
a sibling internal `open-world` workspace, 2026-09-01 -- that
project independently root-caused and fixed the exact jerkiness this bridge
hit on real hardware the same day (see `docs/TRI_LBMFT_EVAL.md:137-223`
there). Diagnosis, in short: pi0.5 re-samples its rng every `infer()` call,
and because these checkpoints emit an ABSOLUTE `xyzrot6g` pose rather than a
delta, that per-call sampling variance lands directly on the commanded
pose -- replaying byte-identical frames through the policy reproduces the
same jitter, so it isn't the robot, the network, or (as this file's
existence implies) client-side decoding. Their fix, proven on 10 real
rollouts (2.4-4.6x wobble reduction, seam 2.0-3.9x, grasps preserved 5/5):
average several independent draws per inference call, optionally blend
overlapping chunks over time (ACT-style temporal ensembling) -- both
routed through the SO(3)-correct mean here -- and never average or
temporally-blend the gripper columns (see module docstring in
pi0_policy_wrapper.py's caller for why: recorded gripper commands are
effectively binary, and averaging draws that disagree about grasp *phase*
produces a mid-value command that appears nowhere in training and can miss
a grasp entirely).

The action layout constants below (R_ROT=3:9, L_ROT=12:18, R_GRIP=18,
L_GRIP=19) were derived independently on that project from 44,738 real
frames and match this bridge's own (HF-README-sourced) decoding exactly --
a useful cross-check that both are right.
"""

import numpy as np

R_XYZ = slice(0, 3)
R_ROT = slice(3, 9)
L_XYZ = slice(9, 12)
L_ROT = slice(12, 18)
R_GRIP = 18
L_GRIP = 19
ROT_SLICES = (R_ROT, L_ROT)
GRIP_COLS = np.array([R_GRIP, L_GRIP], dtype=np.int64)
ACTION_DIM = 20


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """``[...,6]`` -> ``[...,3,3]`` via Gram-Schmidt (Zhou et al. 2019)."""
    r = np.asarray(r6, dtype=np.float64)
    a1, a2 = r[..., :3], r[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:  # noqa: N803
    """``[...,3,3]`` -> ``[...,6]``; inverse of :func:`rot6d_to_matrix`."""
    R = np.asarray(R, dtype=np.float64)  # noqa: N806
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def blend_actions20(actions: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted mean of N 20-d ``xyzrot6g`` actions, rotations averaged ON SO(3).

    For xyz and the grippers a plain weighted mean is correct. For the two
    rot6d blocks it is NOT: averaging the six numbers column-wise leaves the
    manifold -- the columns shrink below unit norm and stop being
    orthogonal -- so each rotation block is converted to a matrix, averaged
    as a true rotation (chordal/quaternion mean via
    `scipy.spatial.transform.Rotation.mean`), and converted back.

    Callers that care about gripper phase (see module docstring) should
    overwrite `out[..., GRIP_COLS]` themselves after calling this --  it
    blends every column including the grippers.
    """
    from scipy.spatial.transform import Rotation

    A = np.asarray(actions, dtype=np.float64).reshape(-1, ACTION_DIM)  # noqa: N806
    if A.shape[0] == 1:
        return A[0].copy()
    w = np.ones(A.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (A.shape[0],):
        raise ValueError(f"weights must be one per action, got {w.shape} for {A.shape[0]}")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"blend weights must be finite and sum > 0, got sum={total}")
    w = w / total

    out = w @ A
    for sl in ROT_SLICES:
        mats = rot6d_to_matrix(A[:, sl])
        out[sl] = matrix_to_rot6d(Rotation.from_matrix(mats).mean(weights=w).as_matrix())
    return out
