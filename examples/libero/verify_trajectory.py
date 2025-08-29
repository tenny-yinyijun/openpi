import os
import h5py
from sympy import fps
import tqdm
import tyro
import imageio
import numpy as np
import dataclasses

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    # file_path: str = "/n/fs/tom-project/papers/openpi/data/libero/0828-test0-2000/trajectories/traj_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.hdf5"#"0.0.0.0"
    file_path: str = "/n/fs/tom-project/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5" 
    output_folder: str = "/n/fs/tom-project/papers/openpi/data/libero/0828-test0-2000/videos-from-traj"
    fps: int = 20

def export_trajectory_video(args: Args) -> None:
    f = h5py.File(args.file_path, "r")["data"]
    # create folder if not exist
    os.makedirs(args.output_folder, exist_ok=True)
    num_success = 0
    for i in tqdm.tqdm(range(len(f))):
        traj = f["demo_" + str(i)]
        agentview_rgb = traj["obs"]["agentview_rgb"]  # [T, H, W, C]
        # agent view is up-side-down
        agentview_rgb = agentview_rgb[:][:, ::-1, :, :]  # flip height dimension
        
        eyeinhand_rgb = traj["obs"]["eye_in_hand_rgb"]  # [T, H, W, C]
        obs_rgb = np.concatenate([agentview_rgb, eyeinhand_rgb], axis=2).astype(np.uint8) # [T, H, W*2, C]
        if traj["dones"][-1] == 1:
            num_success += 1
        video_path = os.path.join(args.output_folder, f"video_{i}.mp4")
        
        # all_obs = np.concatenate(all_obs, axis=0)
        imageio.mimsave(video_path, obs_rgb, fps=args.fps)
    print(f"Successfully saved video to: {args.output_folder} | success rate: {num_success / len(f):.2f}")
    return True

if __name__ == "__main__":
    tyro.cli(export_trajectory_video)
