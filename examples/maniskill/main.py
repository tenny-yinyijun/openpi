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
    env_id: str = "PickPlaceYCBStaticCamera-v1"
    num_evals: int = 5
    task_description: str = "Put the banana into the bowl"
    max_timestep: int = 300  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/maniskill/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_maniskill(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    # TODO input task instruction
    task_description = args.task_description
    env_id = args.env_id
    num_evals = args.num_evals
    
    logging.info(f"Evaluating task: {task_description}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for eval_idx in range(num_evals):
        # initialize environment
        env = _get_maniskill_env(env_id)
        
        obs, info = env.reset(seed=args.seed + eval_idx)
        action_plan = collections.deque()
        
        t = 0
        replay_images = []
        
        # start roll-out
        for step in tqdm.tqdm(range(args.max_timestep)):
            try:
            
                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).cpu().numpy())
                wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"].squeeze(0).cpu().numpy())
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                # Save preprocessed image for replay video
                print("img shape:", img.shape, wrist_img.shape)
                
                # concatenate two views side-by-side
                concat_img = np.concatenate([img, wrist_img], axis=1)
                replay_images.append(concat_img)

                if not action_plan:
                    agentview_rgb = obs["sensor_data"]["base_camera"]["rgb"]
                    eyeinhand_rgb = obs["sensor_data"]["hand_camera"]["rgb"]
                    print(agentview_rgb.shape, eyeinhand_rgb.shape)
                    # remove batch dimension
                    agentview_rgb = agentview_rgb.squeeze(0)
                    eyeinhand_rgb = eyeinhand_rgb.squeeze(0)
                    agent_state = obs["agent"]["qpos"][0]
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            agentview_rgb.cpu().numpy(), 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(eyeinhand_rgb.cpu().numpy(), 224, 224),
                        "observation/joint_position": agent_state[:7].cpu().numpy(), # 7d
                        "observation/gripper_position": agent_state[7:8].cpu().numpy(),
                        "prompt": task_description,
                    }

                    # Query model to get action
                    action_chunk = client.infer(request_data)["actions"]
                    print("predicted action chunk:", action_chunk)
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()

                print("action:", action)
                # Execute action in environment
                obs, reward, done, truncated, info = env.step(torch.tensor(action).unsqueeze(0).to(obs["agent"]["qpos"].device))
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1


            except Exception as e:
                logging.error(f"Caught exception: {e}")
                import traceback
                traceback.print_exc()
                break
        num_evals += 1

        # Save a replay video of the episode
        suffix = "success" if done else "failure"
        # task_segment = task_description.replace(" ", "_")
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"rollout_{eval_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=10,
        )
        
        print(f"saved video to {pathlib.Path(args.video_out_path) / f'rollout_{eval_idx}_{suffix}.mp4'}")

        # Log current results
        logging.info(f"Success: {done}")
        # logging.info(f"# successes: {total_successes} ({total_successes / num_evals * 100:.1f}%)")

def _get_maniskill_env(env_id):
    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode="pd_joint_vel",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="rt"),  # default, rt
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="auto" # auto, cpu, gpu
    )
    return env

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_maniskill)
