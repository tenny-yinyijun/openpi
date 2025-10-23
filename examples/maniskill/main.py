import collections
import dataclasses
import logging
import math
import os
import pathlib
import time

import imageio

import mani_skill.envs  # triggers the registration
import gymnasium as gym

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import torch

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "neu325"#"0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # Experiment settings
    #################################################################################################################
    env_id: str = "PushObjectDroid-v1"
    num_evals: int = 10
    # task_description: str = "Put the banana into the bowl"
    max_timestep: int = 200  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/maniskill/videos"  # Path to save videos

    seed: int = 5000  # Random Seed (for reproducibility)


def eval_maniskill(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    # TODO input task instruction
    # task_description = args.task_description
    env_id = args.env_id
    num_evals = args.num_evals
    
    # logging.info(f"Evaluating task: {task_description}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for eval_idx in range(num_evals):
        # initialize environment
        env = _get_maniskill_env(env_id)
        
        obs, info = env.reset(seed=args.seed + eval_idx)
        
        task_description = env.instruction
        logging.info(f"Evaluating task: {task_description}")
        action_plan = collections.deque()
        
        t = 0
        replay_images = []
        
        # start roll-out
        for step in tqdm.tqdm(range(args.max_timestep)):
            try:
            
                # Get preprocessed image
                img = np.ascontiguousarray(obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).cpu().numpy())
                wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"].squeeze(0).cpu().numpy())
                img = image_tools.convert_to_uint8(img)
                wrist_img = image_tools.convert_to_uint8(wrist_img)

                # concatenate two views side-by-side
                concat_img = np.concatenate([img, wrist_img], axis=1)
                replay_images.append(concat_img)

                if not action_plan:
                    agentview_rgb = obs["sensor_data"]["base_camera"]["rgb"]
                    eyeinhand_rgb = obs["sensor_data"]["hand_camera"]["rgb"]
                    # remove batch dimension
                    agentview_rgb = agentview_rgb.squeeze(0)
                    eyeinhand_rgb = eyeinhand_rgb.squeeze(0)
                    agent_state = obs["agent"]["qpos"][0]
                    request_data = {
                        "observation/image": image_tools.resize_with_pad(
                            agentview_rgb.cpu().numpy(), 180, 320
                        ),
                        "observation/wrist_image": image_tools.resize_with_pad(eyeinhand_rgb.cpu().numpy(), 180, 320),
                        "observation/state": agent_state.cpu().numpy(), # 7d
                        "prompt": task_description,
                    }

                    # Query model to get action
                    result = client.infer(request_data)
                    action_chunk = result["actions"]
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                
                obs, reward, done, truncated, info = env.step(torch.tensor(action).unsqueeze(0).to(obs["agent"]["qpos"].device))
                # if done:
                #     task_successes += 1
                #     total_successes += 1
                    # break
                t += 1


            except Exception as e:
                logging.error(f"Caught exception: {e}")
                import traceback
                traceback.print_exc()
                break
        num_evals += 1

        # Save a replay video of the episode
        # suffix = "success" if done else "failure"
        # task_segment = task_description.replace(" ", "_")
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"{env_id}_rollout_{eval_idx}.mp4", #_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_maniskill)
