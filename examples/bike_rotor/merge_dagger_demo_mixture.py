"""Merge the DAgger takes with an equal number of demonstration frames.

openpi resolves data by train-config name and LeRobotBikeRotorDataConfig takes a SINGLE
repo_id (see DAGGER_PI05.md S6: "there is also no data mixing support"), so a 50/50 mix
has to exist as one dataset on disk rather than as sampling weights at train time.

Corrections-only fine-tuning risks forgetting; pairing each correction frame with a
demonstration frame is the cheap guard.
"""
import argparse, json, shutil, numpy as np, torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

RESIZE_H, RESIZE_W, STATE_DIM, ACTION_DIM = 224, 224, 16, 20
SLOTS = ["observation.images.base", "observation.images.left_wrist",
         "observation.images.right_wrist"]

def to_hwc_u8(t):
    a = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3):        # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    return a

def episode_rows(ds, ep):
    lo = int(ds.episode_data_index["from"][ep]); hi = int(ds.episode_data_index["to"][ep])
    return lo, hi

def copy_episode(src, out, ep, task_override=None):
    lo, hi = episode_rows(src, ep)
    for i in range(lo, hi):
        s = src[i]
        frame = {k: to_hwc_u8(s[k]) for k in SLOTS}
        frame["observation.state"] = np.asarray(s["observation.state"], dtype=np.float32)
        frame["actions"] = np.asarray(s["actions"], dtype=np.float32)
        frame["task"] = task_override or src.meta.tasks[int(s["task_index"])]
        out.add_frame(frame)
    out.save_episode()
    return hi - lo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dagger-repo", default="tri/bike_rotor_dagger_v1")
    ap.add_argument("--demo-repo", default="tri/bike_rotor_cartesian")
    ap.add_argument("--out-repo", default="tri/bike_rotor_dagger_mix50")
    ap.add_argument("--demo-frac", type=float, default=1.0,
                    help="demo frames per dagger frame (1.0 = 50/50)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-eps", type=int, default=0, help="smoke test: cap episodes per source")
    a = ap.parse_args()

    dag = LeRobotDataset(a.dagger_repo, video_backend="pyav")
    demo = LeRobotDataset(a.demo_repo, video_backend="pyav")
    n_dag_eps = dag.meta.total_episodes
    n_dag_rows = dag.meta.total_frames
    target_demo_rows = int(round(n_dag_rows * a.demo_frac))
    print(f"  dagger: {n_dag_eps} eps / {n_dag_rows} rows")
    print(f"  demo  : {demo.meta.total_episodes} eps / {demo.meta.total_frames} rows")
    print(f"  target demo rows: {target_demo_rows}")

    out_path = HF_LEROBOT_HOME / a.out_repo
    if out_path.exists():
        shutil.rmtree(out_path)
    img = {"dtype": "video", "shape": (RESIZE_H, RESIZE_W, 3),
           "names": ["height", "width", "channel"]}
    out = LeRobotDataset.create(
        repo_id=a.out_repo, robot_type="bimanual_panda", fps=dag.meta.fps,
        features={**{s: img for s in SLOTS},
                  "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
                  "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]}},
        image_writer_threads=10, image_writer_processes=5)

    manifest, dag_rows, demo_rows = [], 0, 0
    eps = range(n_dag_eps) if not a.limit_eps else range(min(a.limit_eps, n_dag_eps))
    for ep in eps:
        n = copy_episode(dag, out, ep)
        dag_rows += n
        manifest.append({"episode_index": len(manifest), "source": "dagger",
                         "src_repo": a.dagger_repo, "src_episode": ep, "rows": n})
        print(f"    dagger ep {ep} -> {n} rows", flush=True)

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(demo.meta.total_episodes)
    for ep in order:
        if demo_rows >= target_demo_rows: break
        if a.limit_eps and len(manifest) - len(list(eps)) >= a.limit_eps: break
        n = copy_episode(demo, out, int(ep))
        demo_rows += n
        manifest.append({"episode_index": len(manifest), "source": "demo",
                         "src_repo": a.demo_repo, "src_episode": int(ep), "rows": n})
        print(f"    demo ep {ep} -> {n} rows (demo total {demo_rows}/{target_demo_rows})", flush=True)

    with open(out_path / "mixture_manifest.json", "w") as f:
        json.dump({"dagger_rows": dag_rows, "demo_rows": demo_rows,
                   "demo_frac_actual": demo_rows / max(1, dag_rows),
                   "seed": a.seed, "episodes": manifest}, f, indent=2)
    print(f"\n  DONE: {len(manifest)} episodes, {dag_rows} dagger + {demo_rows} demo rows "
          f"= {dag_rows + demo_rows} ({100*demo_rows/(dag_rows+demo_rows):.1f}% demo)")
    print(f"  {out_path}")

main()
