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

    #################################################################################################################
    # World Model settings
    #################################################################################################################
    wm_id: str = "stack-all-test"
    
    #################################################################################################################
    # Experiment settings
    #################################################################################################################
    split: str = "train"
    num_eval_per_seed : int = 5
    num_seed: int = 5
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
    num_eval_per_seed = args.num_eval_per_seed
    num_seed = args.num_seed
    num_evals = num_eval_per_seed * num_seed
    
    environment_seeds = AVAILABLE_SEEDS[env_id][args.split]
    
    # parse world model config
    wm_id = args.wm_id
    assert wm_id in WORLD_MODELS, f"World Model ID {wm_id} not found in available checkpoints: {WORLD_MODELS.keys()}"
    wm_obs_type = WORLD_MODELS[wm_id]["obs_type"]
    wm_ckpt = WORLD_MODELS[wm_id]["ckpt"]
    wm_config_dir = os.path.dirname(wm_ckpt)
    wm_config_path = os.path.join(wm_config_dir, "config.yaml")

    cfg = OmegaConf.load(wm_config_path)
    # chunk_size = cfg.n_frames # TODO
    chunk_size = 5
    chunk_freq = cfg.frame_skip
    n_actions = chunk_size * chunk_freq

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for eval_idx in range(num_evals):
        # initialize environment
        eval_seed = environment_seeds[(eval_idx // num_eval_per_seed) % len(environment_seeds)]
        print(f"Starting evaluation {eval_idx+1}/{num_evals} with seed {eval_seed}...")

        wm_env = _get_wm_env(wm_obs_type, wm_config_path, wm_ckpt)
        
        # initialize env
        mani_env = _get_maniskill_env(env_id)
        mani_replay_env = _get_maniskill_env(env_id)
        
        # reset
        obs_gt, info_gt = mani_env.reset(seed=eval_seed)
        obs, info = mani_replay_env.reset(seed=eval_seed)
        
        task_description = mani_env.instruction
        logging.info(f"Evaluating task: {task_description}")
        
        # initialize observation
        initial_agentview_rgb = obs["sensor_data"]["base_camera"]["rgb"]
        initial_agentview_rgb2 = obs["sensor_data"]["base_camera_2"]["rgb"]
        initial_eyeinhand_rgb = obs["sensor_data"]["hand_camera"]["rgb"]
        
        # remove batch dimension
        initial_agentview_rgb = initial_agentview_rgb.squeeze(0)
        initial_agentview_rgb2 = initial_agentview_rgb2.squeeze(0)
        initial_eyeinhand_rgb = initial_eyeinhand_rgb.squeeze(0)
        initial_agent_state = obs["agent"]["qpos"][0]

        wm_env.initialize_from_maniskill(initial_agentview_rgb, initial_agentview_rgb2, initial_eyeinhand_rgb)

        t_gt = 0
        gt_images = []
        replay_gt_images = []
        replay_wm_images = []
        
        gt_action_list = []
        wm_action_list = []
        gt_state_list = []
        wm_state_list = []
        
        # ========== SAVE VIDEOS ==========

        # start rollout in maniskill
        agentview_rgb = initial_agentview_rgb
        eyeinhand_rgb = initial_eyeinhand_rgb
        agent_state = initial_agent_state
        
        while t_gt < args.max_timestep:

            # policy infer
            request_data = {
                "observation/image": image_tools.resize_with_pad(agentview_rgb.cpu().numpy(), 180, 320),
                "observation/wrist_image": image_tools.resize_with_pad(eyeinhand_rgb.cpu().numpy(), 180, 320),
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
            
            downsampled_actions = action_to_execute[::chunk_freq]

            gt_action_list.extend(downsampled_actions.tolist())

            # maniskill step
            for a in downsampled_actions:
                # save gt observation
                img = np.ascontiguousarray(obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).cpu().numpy())
                img2 = np.ascontiguousarray(obs["sensor_data"]["base_camera_2"]["rgb"].squeeze(0).cpu().numpy())
                wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"].squeeze(0).cpu().numpy())
                img = image_tools.convert_to_uint8(img)
                img2 = image_tools.convert_to_uint8(img2)
                wrist_img = image_tools.convert_to_uint8(wrist_img)

                # concatenate two views side-by-side
                concat_img = np.concatenate([img, img2, wrist_img], axis=0)
                gt_images.append(concat_img)
                
                # state
                gt_state_list.append(obs["agent"]["qpos"][0].cpu().numpy().tolist())

                # step environment
                obs, reward, done, truncated, info = mani_env.step(torch.tensor(a).unsqueeze(0).to(obs["agent"]["qpos"].device))

            # update observation
            agentview_rgb = obs["sensor_data"]["base_camera"]["rgb"].squeeze(0)
            eyeinhand_rgb = obs["sensor_data"]["hand_camera"]["rgb"].squeeze(0)
            agent_state = obs["agent"]["qpos"][0]
            
            t_gt += n_actions
            
        t = 0
        agentview_rgb = initial_agentview_rgb
        eyeinhand_rgb = initial_eyeinhand_rgb
        agent_state = initial_agent_state
        
        # start roll-out in world model
        while t < args.max_timestep:
            # if datatype doesn't match, convert
            if agentview_rgb.dtype != torch.uint8:
                agentview_arr = (agentview_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                eyeinhand_arr = (eyeinhand_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            else:
                agentview_arr = agentview_rgb.cpu().numpy()
                eyeinhand_arr = eyeinhand_rgb.cpu().numpy()
            
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
            
            wm_action_list.extend(downsampled_actions.tolist())

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
                
                wm_state_list.append(obs["agent"]["qpos"][0].cpu().numpy().tolist())
                
                # step environment
                obs, reward, done, truncated, info = mani_replay_env.step(torch.tensor(a).unsqueeze(0).to(obs["agent"]["qpos"].device))
            
            # upate obs
            agentview_rgb = wm_result_dict["base_camera"]
            eyeinhand_rgb = wm_result_dict["hand_camera"]
            agent_state = obs["agent"]["qpos"][0]
            
            replay_wm_images.append(wm_result_dict["generated_frames"])  # exclude last frame which is next obs
            
            t += n_actions
            
        num_evals += 1
        
        # ========== SAVE VIDEOS ==========
        
        replay_wm_images_concat = torch.cat(replay_wm_images, dim=0)
        replay_wm_frames = list(torch.unbind(replay_wm_images_concat, dim=0))
        
        print("Length of wm replay frames:", len(replay_wm_frames))
        print("Length of gt replay frames:", len(replay_gt_images))
        print("Length of gt images:", len(gt_images))
        
        # subsample gt_images and replay_gt_images to match wm frames
        subsample_factor = math.ceil(len(replay_gt_images) / len(replay_wm_frames))
        replay_gt_images = replay_gt_images[::subsample_factor]
        # gt_images = gt_images[::subsample_factor]
        
        print("After subsampling:")
        print("Length of gt frames:", len(gt_images))
        print("Length of gt replay frames:", len(replay_gt_images))
        
        # make sure the eval_seed directory exists
        eval_directory = pathlib.Path(args.video_out_path) / f"seed{eval_seed}"
        eval_directory.mkdir(parents=True, exist_ok=True)
        
        
        # ========== SAVE TRAJECTORIES ==========
        
        # make sure trajectory directory exists
        traj_directory = eval_directory / "trajectories"
        traj_directory.mkdir(parents=True, exist_ok=True)
        
        # save gt
        data_json_path = traj_directory / f"{eval_idx}-gt.json"
        json_data = {
            "state": gt_state_list,
            "actions": gt_action_list,
            "steps": t,
            "env_seed": eval_seed,
        }
        with open(data_json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
        print(f"Saved gt data JSON to {data_json_path}")
        
        # save wm
        data_json_path = traj_directory / f"{eval_idx}-wm.json"
        json_data = {
            "state": wm_state_list,
            "actions": wm_action_list,
            "steps": t,
            "env_seed": eval_seed,
        }
        with open(data_json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
        print(f"Saved wm data JSON to {data_json_path}")
        
        # ========== SAVE VIDEOS ==========
        
        # convert frames to uint8 and save
        uint_wm_frames = []
        for frame in replay_wm_frames:
            frame = (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            uint_wm_frames.append(frame)
        
        # save gt video
        imageio.mimwrite(
            eval_directory / f"{env_id}_rollout_{eval_idx}-gt.mp4",
            [np.asarray(x) for x in gt_images],
            fps=10
        )
        
        # save wm generated video
        imageio.mimwrite(
            eval_directory / f"{env_id}_rollout_{eval_idx}-wm.mp4",
            uint_wm_frames,
            fps=10
        )

        # save reply gt video
        imageio.mimwrite(
            eval_directory / f"{env_id}_rollout_{eval_idx}-gtreplay.mp4", #_{suffix}.mp4",
            [np.asarray(x) for x in replay_gt_images],
            fps=10,
        )

        print(f"saved video to {eval_directory / f'{env_id}_rollout_{eval_idx}.mp4'}")

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
