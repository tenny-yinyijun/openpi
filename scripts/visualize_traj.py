import h5py
import numpy as np
import imageio

input_path="/n/fs/iromdata/modified_libero/libero_90_no_noops/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5"

output_video_name="KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.mp4"




output_folder="/n/fs/tom-project/video_models/outputs/gt"

output_path = output_folder + "/" + output_video_name

f = h5py.File(input_path, "r")["data"]
traj = f["demo_0"]

agentview_rgb = traj["obs"]["agentview_rgb"]  # [T, H, W, C]
eyeinhand_rgb = traj["obs"]["eye_in_hand_rgb"]  # [T, H, W, C]
obs_rgb = np.concatenate([agentview_rgb, eyeinhand_rgb], axis=1).astype(np.uint8)  # [T, H*2, W, C]
imageio.mimsave(output_path, obs_rgb, fps=30)
print(f"Successfully saved video to: {output_path}")