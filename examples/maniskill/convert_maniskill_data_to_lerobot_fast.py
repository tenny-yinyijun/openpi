"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import h5py
import json

import numpy as np
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

# REPO_NAME = "tennyyin/maniskill-all-jointpos"  # Name of the output dataset, also used for the Hugging Face Hub
# MANISKILL_DATASETS = {
#     "PickPlaceDroid-v1": "/n/fs/iromdata/pi_finetune_data/maniskill/PickPlaceDroid-v1/n500_20251021-010733/trajectory.h5",
#     "PickPlaceNextDroid-v1": "/n/fs/iromdata/pi_finetune_data/maniskill/PickPlaceNextDroid-v1/n500_20251021-010734/trajectory.h5",
#     "PickPlaceOutDroid-v1": "/n/fs/iromdata/pi_finetune_data/maniskill/PickPlaceOutDroid-v1/n500_20251021-010734/trajectory.h5",
#     "PushObjectDroid-v1": "/n/fs/iromdata/pi_finetune_data/maniskill/PushObjectDroid-v1/n500_20251021-010733/trajectory.h5"
# }

# REPO_NAME = "tennyyin/mani-stack-200"  # Name of the output dataset, also used for the Hugging Face Hub
# MANISKILL_DATASETS = {
#     "StackCube-v1": "/n/fs/iromdata/pi_finetune_data/maniskill/StackCube-v1/n200_20251031-193207/trajectory.h5"
# }

REPO_NAME = "tennyyin/mani-stack-400"  # Name of the output dataset, also used for the Hugging Face Hub
MANISKILL_DATASETS = {
    "StackCube-v1": "/n/fs/iromdata/pi_finetune_data/maniskill/StackCube-v1/n400_20251101-161512/trajectory.h5"
}

def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def main():
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,
        features={
            "image": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name, trajectory_path in MANISKILL_DATASETS.items():
        print("Processing dataset:", raw_dataset_name)
        raw_dataset = h5py.File(trajectory_path, 'r')
        annotations = json.load(open(trajectory_path.replace('.h5', '.json'), 'r'))["episodes"]

        for traj_id, episode in raw_dataset.items():
            i = int(traj_id.split('_')[-1])
            instr = annotations[i]["instruction"]

            rgb_base = episode["obs"]["sensor_data"]["base_camera"]["rgb"][:]
            rgb_wrist = episode["obs"]["sensor_data"]["hand_camera"]["rgb"][:]
            qpos = episode["obs"]["agent"]["qpos"][:]
            actions = episode["actions"][:]

            diff = np.abs(np.diff(actions, axis=0))
            keep = np.concatenate([[True], np.any(diff > 1e-3, axis=1)])

            frames = [
                {
                    "image": rgb_base[j],
                    "wrist_image": rgb_wrist[j],
                    "state": qpos[j],
                    "actions": actions[j],
                    "task": instr,
                }
                for j in np.where(keep)[0]
            ]

            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()

if __name__ == "__main__":
    tyro.cli(main)
