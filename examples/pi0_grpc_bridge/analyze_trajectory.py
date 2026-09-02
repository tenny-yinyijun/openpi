"""One-off analysis of a pi0_policy_server.py trajectory log (see
_log_step in pi0_policy_wrapper.py). Plots measured vs. commanded pose per
axis for both arms, with chunk boundaries marked, to diagnose jerkiness /
oscillation reported on real hardware.

Usage: python analyze_trajectory.py <path-to-.jsonl> [out.png]
"""

import json
import sys

import matplotlib as mpl

# Must precede the pyplot import: this runs headless on the station.
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def seam_report(rows, cadence=8):
    """The chunk-boundary discontinuity metric -- the one that caught the
    2026-09-01 backward-jerk bug, and the one to re-run after any change to
    prefetch/latency handling.

    Two numbers per arm:

    * How much of the boundary jump points AGAINST the direction the arm was
      already travelling, and what fraction of it undoes the preceding
      `cadence` ticks of motion. A healthy seam is small and unsigned; the
      bug showed 86-93% of boundaries moving backward by 0.6-0.9x of the
      previous 7 ticks of travel.
    * Where the commanded pose sits relative to the measured one, mid-chunk
      versus on the fresh tick. Mid-chunk the command leads (the controller
      is chasing it); if it flips to trailing on fresh ticks, chunks are
      being entered at a row that is in the past -- raise
      --action-latency-ticks, or check the prefetch anchoring.

    Logs can contain several episodes/clients, so this splits on client
    first: mixing them invents a boundary at each splice.
    """
    by_client = {}
    for r in rows:
        by_client.setdefault(r["client"], []).append(r)

    for client, rs in by_client.items():
        fresh_idx = [i for i, r in enumerate(rs) if r["is_fresh_chunk"]]
        print(f"\n--- seam report: client {client[:8]} ({len(rs)} steps, {len(fresh_idx)} boundaries) ---")
        if len(rs) < 2 * cadence:
            print("  (too short to be meaningful)")
            continue
        for side in ("right", "left"):
            cmd = np.array([r["commanded"][f"{side}_pos"] for r in rs])
            meas = np.array([r["measured"][f"{side}_pos"] for r in rs])
            along, jump, travel, lead_mid, lead_fresh = [], [], [], [], []
            for i in fresh_idx:
                if i < cadence:
                    continue
                prev = cmd[i - 1] - cmd[i - cadence]
                norm = np.linalg.norm(prev)
                if norm < 1e-6:
                    continue
                unit = prev / norm
                along.append(float((cmd[i] - cmd[i - 1]) @ unit))
                jump.append(float(np.linalg.norm(cmd[i] - cmd[i - 1])))
                travel.append(float(norm))
                lead_mid.append(float((cmd[i - 1] - meas[i - 1]) @ unit))
                lead_fresh.append(float((cmd[i] - meas[i]) @ unit))
            if not along:
                continue
            along = np.array(along)
            print(
                f"  {side}: |jump|={np.mean(jump) * 1000:5.1f}mm   "
                f"along-travel={along.mean() * 1000:+6.1f}mm   "
                f"backward at {(along < 0).mean() * 100:3.0f}% of seams   "
                f"undoes {np.mean(-along / np.array(travel)):+.2f}x of prior "
                f"{cadence} ticks"
            )
            print(
                f"        cmd-vs-measured lead: mid-chunk="
                f"{np.mean(lead_mid) * 1000:+6.1f}mm  "
                f"on fresh tick={np.mean(lead_fresh) * 1000:+6.1f}mm"
            )

        dt = np.diff([r["t"] for r in rs])
        is_fresh = np.array([r["is_fresh_chunk"] for r in rs])[1:]
        if is_fresh.any() and (~is_fresh).any():
            print(
                f"  tick dt: normal={dt[~is_fresh].mean():.3f}s  "
                f"fresh-chunk={dt[is_fresh].mean():.3f}s  "
                "(a fresh-tick spike means the prefetch lead is too short)"
            )


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/trajectory_plot.png"
    rows = load(path)
    t0 = rows[0]["t"]
    t = np.array([r["t"] - t0 for r in rows])
    chunk_id = np.array([r["chunk_id"] for r in rows])
    fresh = np.array([r["is_fresh_chunk"] for r in rows])

    def arr(key, side, axis):
        return np.array([r[key][f"{side}_pos"][axis] for r in rows])

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    axis_names = ["x", "y", "z"]
    labels = ["right", "left"]

    for i, axis_name in enumerate(axis_names):
        ax = axes[i]
        for side in labels:
            ax.plot(t, arr("measured", side, i), label=f"{side} measured", linewidth=1.5)
            ax.plot(
                t,
                arr("commanded", side, i),
                label=f"{side} commanded",
                linewidth=1,
                linestyle="--",
            )
        for boundary_t in t[fresh]:
            ax.axvline(boundary_t, color="gray", alpha=0.3, linewidth=0.8)
        ax.set_ylabel(f"{axis_name} (m)")
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(t, [r["measured"]["right_gripper"] for r in rows], label="right gripper measured")
    ax.plot(t, [r["commanded"]["right_gripper"] for r in rows], "--", label="right gripper cmd")
    ax.plot(t, [r["measured"]["left_gripper"] for r in rows], label="left gripper measured")
    ax.plot(t, [r["commanded"]["left_gripper"] for r in rows], "--", label="left gripper cmd")
    for boundary_t in t[fresh]:
        ax.axvline(boundary_t, color="gray", alpha=0.3, linewidth=0.8)
    ax.set_ylabel("gripper (m)")
    ax.set_xlabel("time (s)   [gray lines = fresh inference call / chunk boundary]")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{path}\n{len(rows)} steps, {chunk_id.max() + 1} chunks")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")

    # Also print a quick numeric summary: per-chunk z range for both arms,
    # and the largest single-tick z jump (measured) with its context.
    print("\n--- per-chunk right-arm z range ---")
    for cid in sorted(set(chunk_id.tolist())):
        mask = chunk_id == cid
        z = arr("measured", "right", 2)[mask]
        print(f"chunk {cid:2d}: n={mask.sum():2d}  z=[{z.min():.4f}, {z.max():.4f}]  range={z.max() - z.min():.4f}")

    print("\n--- largest single-tick measured-z jumps (right arm) ---")
    z = arr("measured", "right", 2)
    dz = np.diff(z)
    order = np.argsort(-np.abs(dz))[:10]
    for idx in order:
        print(
            f"t={t[idx]:.3f}->{t[idx + 1]:.3f}  dz={dz[idx]:+.4f}  "
            f"chunk {chunk_id[idx]}->{chunk_id[idx + 1]}  fresh_next={fresh[idx + 1]}"
        )

    seam_report(rows)


if __name__ == "__main__":
    main()
