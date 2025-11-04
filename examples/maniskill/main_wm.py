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

import mani_skill.envs  # triggers the registration
import gymnasium as gym

import numpy as np
from dynuq.env.wm_env import WorldModelEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import torch

from omegaconf import OmegaConf

with open("examples/maniskill/available_checkpoints.json", "r") as f:
    AVAILABLE_CHECKPOINTS = json.load(f)
    POLICIES = AVAILABLE_CHECKPOINTS["policy_checkpoints"]
    WORLD_MODELS = AVAILABLE_CHECKPOINTS["world_model_checkpoints"]
    AVAILABLE_SEEDS = AVAILABLE_CHECKPOINTS["available_seeds"]

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "neu325"#"0.0.0.0"
    port: int = 8000
    resize_size: int = 224

    #################################################################################################################
    # Environment settings
    #################################################################################################################
    env_id: str = "StackCube-v1"
    gt: bool = True # whether to display ground-truth states side-by-side
    
    #################################################################################################################
    # World Model settings
    #################################################################################################################
    wm_id: str = "stack-test"
    
    #################################################################################################################
    # Experiment settings
    #################################################################################################################
    split: str = "train"
    num_evals: int = 10
    max_timestep: int = 200 #200

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/maniskill/wm"  # Path to save videos

    seed: int = 5000  # Random Seed (for reproducibility)


def eval_maniskill(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    # parse environment config
    env_id = args.env_id
    num_evals = args.num_evals
    
    environment_seeds = AVAILABLE_SEEDS[env_id][args.split]
    
    # parse world model config
    wm_id = args.wm_id
    assert wm_id in WORLD_MODELS, f"World Model ID {wm_id} not found in available checkpoints: {WORLD_MODELS.keys()}"
    wm_obs_type = WORLD_MODELS[wm_id]["obs_type"]
    wm_ckpt = WORLD_MODELS[wm_id]["ckpt"]
    
    # TODO hard code wm config path: under same directory as checkpoint
    wm_config_dir = os.path.dirname(wm_ckpt)
    wm_config_path = os.path.join(wm_config_dir, "config.yaml")

    cfg = OmegaConf.load(wm_config_path)
    chunk_size = cfg.n_frames
    chunk_freq = cfg.frame_skip
    n_actions = chunk_size * chunk_freq

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for eval_idx in range(num_evals):
        # initialize environment
        eval_seed = environment_seeds[eval_idx % len(environment_seeds)]
        wm_env = _get_wm_env(wm_obs_type, wm_config_path, wm_ckpt)
        mani_env = _get_maniskill_env(env_id)
        obs, info = mani_env.reset(seed=eval_seed)
        
        task_description = mani_env.instruction
        logging.info(f"Evaluating task: {task_description}")
        
        # get obs dtype
        tmp_rgb = obs["sensor_data"]["base_camera"]["rgb"]
        
        # initialize observation
        agentview_rgb = obs["sensor_data"]["base_camera"]["rgb"]
        agentview_rgb2 = obs["sensor_data"]["base_camera_2"]["rgb"]
        eyeinhand_rgb = obs["sensor_data"]["hand_camera"]["rgb"]
        
        # remove batch dimension
        agentview_rgb = agentview_rgb.squeeze(0)
        agentview_rgb2 = agentview_rgb2.squeeze(0)
        eyeinhand_rgb = eyeinhand_rgb.squeeze(0)
        agent_state = obs["agent"]["qpos"][0]

        wm_env.initialize_from_maniskill(agentview_rgb, agentview_rgb2, eyeinhand_rgb)

        t = 0
        replay_gt_images = []
        replay_wm_images = []

        # start roll-out
        while t < args.max_timestep:
            # if datatype doesn't match, convert
            if agentview_rgb.dtype != torch.uint8:
                agentview_arr = (agentview_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                eyeinhand_arr = (eyeinhand_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            else:
                agentview_arr = agentview_rgb.cpu().numpy()
                eyeinhand_arr = eyeinhand_rgb.cpu().numpy()
                
            imageio.imwrite("agentview.png", agentview_arr)
            imageio.imwrite("eyeinhand.png", eyeinhand_arr)
            
            # policy infer
            request_data = {
                "observation/image": image_tools.resize_with_pad(agentview_arr, 180, 320),
                "observation/wrist_image": image_tools.resize_with_pad(eyeinhand_arr, 180, 320),
                "observation/state": agent_state.cpu().numpy(), # 7d
                "prompt": task_description,
            }
            
            # Query model to get action
            result = client.infer(request_data)
            action_chunk = result["actions"]
            assert (
                len(action_chunk) >= n_actions
            ), f"We want to replan every {n_actions} steps, but policy only predicts {len(action_chunk)} steps."
            action_to_execute = action_chunk[: n_actions]
            
            # wm step
            downsampled_actions = action_to_execute[::chunk_freq]

            wm_result_dict = wm_env.step(downsampled_actions)
            
            # maniskill step
            for a in action_to_execute:
                # save gt observation
                img = np.ascontiguousarray(obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).cpu().numpy())
                img2 = np.ascontiguousarray(obs["sensor_data"]["base_camera_2"]["rgb"].squeeze(0).cpu().numpy())
                wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"].squeeze(0).cpu().numpy())
                img = image_tools.convert_to_uint8(img)
                img2 = image_tools.convert_to_uint8(img2)
                wrist_img = image_tools.convert_to_uint8(wrist_img)

                # concatenate two views side-by-side
                concat_img = np.concatenate([img, img2, wrist_img], axis=0)
                replay_gt_images.append(concat_img)
                
                # step environment
                obs, reward, done, truncated, info = mani_env.step(torch.tensor(a).unsqueeze(0).to(obs["agent"]["qpos"].device))
            
            # upate obs
            agentview_rgb = wm_result_dict["base_camera"]
            eyeinhand_rgb = wm_result_dict["hand_camera"]
            agent_state = obs["agent"]["qpos"][0]
            
            replay_wm_images.append(wm_result_dict["generated_frames"][:-1])  # exclude last frame which is next obs
            
            t += n_actions
                
        num_evals += 1
        
        replay_wm_images_concat = torch.cat(replay_wm_images, dim=0)
        replay_wm_frames = list(torch.unbind(replay_wm_images_concat, dim=0))
        
        print("Length of wm replay frames:", len(replay_wm_frames))
        print("Length of gt replay frames:", len(replay_gt_images))
        
        # subsample replay frames to match lengths
        if len(replay_wm_frames) > len(replay_gt_images):
            factor = len(replay_wm_frames) / len(replay_gt_images)
            indices = [int(i * factor) for i in range(len(replay_gt_images))]
            replay_wm_frames = [replay_wm_frames[i] for i in indices]
        elif len(replay_gt_images) > len(replay_wm_frames):
            factor = len(replay_gt_images) / len(replay_wm_frames)
            indices = [int(i * factor) for i in range(len(replay_wm_frames))]
            replay_gt_images = [replay_gt_images[i] for i in indices]
        print("After subsampling:")
        print("Length of wm replay frames:", len(replay_wm_frames))
        print("Length of gt replay frames:", len(replay_gt_images))
        
        # Save a replay video of the episode
        
        # convert frames to uint8 and save
        uint_wm_frames = []
        for frame in replay_wm_frames:
            frame = (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            uint_wm_frames.append(frame)
        
        # save wm generated video
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"{env_id}_rollout_{eval_idx}-wm.mp4",
            uint_wm_frames,
            fps=10
        )

        # save gt video
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"{env_id}_rollout_{eval_idx}-gt.mp4", #_{suffix}.mp4",
            [np.asarray(x) for x in replay_gt_images],
            fps=10,
        )

        print(f"saved video to {pathlib.Path(args.video_out_path) / f'{env_id}_rollout_{eval_idx}.mp4'}")

def _get_maniskill_env(env_id):
    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="rt"),  # default, rt
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="cpu" # auto, cpu, gpu
    )
    return env

def _get_wm_env(obs_type, config_path, checkpoint_path):
    env = WorldModelEnv(
        obs_type=obs_type,
        config_path=config_path,
        checkpoint_path=checkpoint_path
    )
    return env

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_maniskill)
