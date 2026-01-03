import collections
import dataclasses
import logging
import math
import os
import pathlib
import time
import json
import imageio

import imageio
import gymnasium as gym

import numpy as np
from models.wm_env import WorldModelEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import torch


H, W = 192, 320
NUM_VIEWS = 3
OUT_H = H * NUM_VIEWS
OUT_W = W

FPS = 10

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "neu325"#"0.0.0.0"
    port: int = 8000

    #################################################################################################################
    # World Model settings
    #################################################################################################################
    wm_checkpoint: str = "/n/fs/iromdata/video_models/Ctrl-World/checkpoints/checkpoint-10000.pt"
    
    #################################################################################################################
    # Experiment settings
    #################################################################################################################
    num_interaction: int = 15
    
    seed: int = 42

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/ctrl-world"  # Path to save videos

    seed: int = 5000  # Random Seed (for reproducibility)


def save_eval_video(video, out_path):
    with imageio.get_writer(
        out_path,
        format='FFMPEG',
        fps=FPS,
        codec="libx264",
        quality=8,          # 0–10, higher is better
        pixelformat="yuv420p",
    ) as writer:

        for t in range(len(video)):
            views = video[t]
            num_frames = views[0].shape[0]

            for k in range(num_frames):
                # take kth frame from each view
                frames = [views[v][k] for v in range(NUM_VIEWS)]

                # vertical concat: (576, 320, 3)
                concat = np.concatenate(frames, axis=0)

                if concat.dtype != np.uint8:
                    concat = np.clip(concat, 0, 255).astype(np.uint8)

                writer.append_data(concat)


def eval_world_model(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    n_actions = 15
    # get world model environment
    wm_env = _get_wm_env(args.wm_checkpoint)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # test_ids = [85]
    # test_ids = [105, 123]
    # test_ids = [344, 314]
    test_ids = [314]

    for eval_idx in range(len(test_ids)):
        # initialize environment
        current_obs, current_state, task_description = wm_env.reset(test_ids[eval_idx])
        logging.info(f"Task={test_ids[eval_idx]}, Instruction={task_description}")

        t = 0
        agentview_rgb = torch.tensor(current_obs[1])
        eyeinhand_rgb = torch.tensor(current_obs[2])
        agent_state = torch.tensor(current_state)
        
        print("Initial agentview_rgb shape:", agentview_rgb.shape)

        # start roll-out in world model
        while t < args.num_interaction:
            # if datatype doesn't match, convert
            if isinstance(agentview_rgb, torch.Tensor):
                if agentview_rgb.dtype != torch.uint8:
                    agentview_arr = (agentview_rgb.clamp(0, 1) * 255).to(torch.uint8)
                    eyeinhand_arr = (eyeinhand_rgb.clamp(0, 1) * 255).to(torch.uint8)
                else:
                    agentview_arr = agentview_rgb.to(torch.uint8)
                    eyeinhand_arr = eyeinhand_rgb.to(torch.uint8)

                # Resize from (192, 320) to (180, 320) before resize_with_pad, matching rollout_interact_pi.py
                assert agentview_arr.shape == torch.Size([192, 320, 3]), f"Image shape should be (192, 320, 3), got {agentview_arr.shape}"
                agentview_arr = torch.nn.functional.interpolate(
                    agentview_arr.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(180, 320),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0).to(torch.uint8)
                eyeinhand_arr = torch.nn.functional.interpolate(
                    eyeinhand_arr.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(180, 320),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0).to(torch.uint8)

                agentview_arr = agentview_arr.cpu().numpy()
                eyeinhand_arr = eyeinhand_arr.cpu().numpy()

                joint_position = agent_state[:7].cpu().numpy()
                gripper_position = agent_state[-1:].cpu().numpy()

            # policy infer
            request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(agentview_arr, 224, 224),
                        "observation/wrist_image_left": image_tools.resize_with_pad(eyeinhand_arr, 224, 224),
                        "observation/joint_position": joint_position, # 7d
                        "observation/gripper_position": gripper_position, # 1d
                        "prompt": task_description,
                    }
            
            # Query model to get action
            result = client.infer(request_data)
            action_chunk = result["actions"]
            assert (
                len(action_chunk) >= n_actions
            ), f"We want to replan every {n_actions} steps, but policy only predicts {len(action_chunk)} steps."
            action_to_execute = action_chunk[: n_actions]

            is_last_interact = False
            if t == args.num_interaction - 1:
                is_last_interact = True
            current_obs, current_state, final_video = wm_env.step(action_to_execute, is_last_interact=is_last_interact)

            # upate obs - wrap in torch.tensor to ensure preprocessing happens
            agentview_rgb = torch.tensor(current_obs[1])
            eyeinhand_rgb = torch.tensor(current_obs[2])
            agent_state = torch.tensor(current_state)
            
            t += 1
            
        # num_evals += 1
        
        # # ========== SAVE VIDEOS ==========
        
        save_eval_video(final_video, os.path.join(args.video_out_path, f"test2{test_ids[eval_idx]}.mp4"))
        

def _get_wm_env(checkpoint_path):
    env = WorldModelEnv(
        wm_ckpt=checkpoint_path
    )
    return env

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_world_model)
