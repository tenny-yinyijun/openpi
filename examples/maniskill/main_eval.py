import collections
import dataclasses
import logging
import math
import os
import pathlib
import time
import json

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
    save_eval_video: bool = False  # Whether to save evaluation videos (e.g., StackCube-v1_rollout_0_success.mp4)
    save_data: bool = False  # Whether to save raw data (states, actions, videos) in JSON format

    seed: int = 5000  # Random Seed (for reproducibility)


def eval_maniskill(args: Args) -> None:
    # Set random seed
    np.random.seed(42)

    # TODO input task instruction
    # task_description = args.task_description
    env_id = args.env_id
    num_evals = args.num_evals

    # logging.info(f"Evaluating task: {task_description}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Create output folders for saving raw data
    if args.save_data:
        data_videos_folder = pathlib.Path(args.video_out_path) / "videos_three_view_vertical"
        data_jsons_folder = pathlib.Path(args.video_out_path) / "json"
        data_videos_folder.mkdir(parents=True, exist_ok=True)
        data_jsons_folder.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Track success metrics
    successes = 0
    total_episodes = 0
    episode_results = []

    for eval_idx in range(num_evals):
        # initialize environment
        env = _get_maniskill_env(env_id)

        env_seed = args.seed + eval_idx
        obs, info = env.reset(seed=env_seed)

        task_description = env.instruction
        logging.info(f"Evaluating task (episode {eval_idx + 1}/{num_evals}, seed={env_seed}): {task_description}")
        action_plan = collections.deque()

        t = 0
        replay_images = []
        episode_success = False
        extra_steps_after_done = None  # Track steps after done=True

        # Data collection for save_data option
        if args.save_data:
            agentview_rgb_frames = []
            base_camera_2_rgb_frames = []
            eyeinhand_rgb_frames = []
            action_list = []
            state_list = []

        # start roll-out
        for step in tqdm.tqdm(range(args.max_timestep), desc=f"Episode {eval_idx + 1}"):
            try:
                # Check if we should stop after extra inferences
                if extra_steps_after_done is not None and extra_steps_after_done >= 20:
                    logging.info(f"Episode {eval_idx + 1}: Stopping after {extra_steps_after_done} additional inferences")
                    break

                # Get preprocessed image
                img = np.ascontiguousarray(obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).cpu().numpy())
                wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"].squeeze(0).cpu().numpy())
                img = image_tools.convert_to_uint8(img)
                wrist_img = image_tools.convert_to_uint8(wrist_img)

                # concatenate two views side-by-side
                concat_img = np.concatenate([img, wrist_img], axis=1)
                replay_images.append(concat_img)

                # Save raw RGB frames for data collection
                if args.save_data:
                    agentview_rgb = obs["sensor_data"]["base_camera"]["rgb"]
                    base_camera_2_rgb = obs["sensor_data"]["base_camera_2"]["rgb"]
                    eyeinhand_rgb = obs["sensor_data"]["hand_camera"]["rgb"]
                    agent_state = obs["agent"]["qpos"][0]

                    agentview_rgb_frames.append(agentview_rgb.squeeze(0).cpu().numpy())
                    base_camera_2_rgb_frames.append(base_camera_2_rgb.squeeze(0).cpu().numpy())
                    eyeinhand_rgb_frames.append(eyeinhand_rgb.squeeze(0).cpu().numpy())
                    state_list.append(agent_state.cpu().numpy().tolist())

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

                # Save action for data collection
                if args.save_data:
                    action_list.append(action.tolist())

                obs, reward, done, truncated, info = env.step(torch.tensor(action).unsqueeze(0).to(obs["agent"]["qpos"].device))

                # Check if task is successful
                if done and extra_steps_after_done is None:
                    episode_success = True
                    extra_steps_after_done = 0
                    logging.info(f"Episode {eval_idx + 1}: SUCCESS at step {step + 1}, continuing for 2 more inferences")

                # Increment counter if we're in extra steps
                if extra_steps_after_done is not None:
                    extra_steps_after_done += 1

                t += 1


            except Exception as e:
                logging.error(f"Caught exception: {e}")
                import traceback
                traceback.print_exc()
                break

        # Update statistics
        total_episodes += 1
        if episode_success:
            successes += 1

        episode_results.append({
            'episode': eval_idx,
            'success': episode_success,
            'steps': t
        })

        # Log episode result
        status = "SUCCESS" if episode_success else "FAILURE"
        logging.info(f"Episode {eval_idx + 1} completed: {status} after {t} steps")

        # Save a replay video of the episode (if enabled)
        if args.save_eval_video:
            suffix = "success" if episode_success else "failure"
            video_filename = f"{env_id}_rollout_{eval_idx}_{suffix}.mp4"
            video_path = pathlib.Path(args.video_out_path) / video_filename

            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            print(f"Saved video to {video_path}")

        # Save raw data if enabled
        if args.save_data:
            # Save video with raw RGB frames concatenated
            data_video_path = data_videos_folder / f"{eval_idx}.mp4"
            agentview_rgb_frames_array = np.stack(agentview_rgb_frames, axis=0)
            base_camera_2_rgb_frames_array = np.stack(base_camera_2_rgb_frames, axis=0)
            eyeinhand_rgb_frames_array = np.stack(eyeinhand_rgb_frames, axis=0)

            # Concatenate camera views vertically: base_camera (top), base_camera_2 (middle), hand_camera (bottom) [T, H*3, W, C]
            obs_rgb = np.concatenate([agentview_rgb_frames_array, base_camera_2_rgb_frames_array, eyeinhand_rgb_frames_array], axis=1).astype(np.uint8)
            imageio.mimsave(str(data_video_path), obs_rgb, fps=30)
            print(f"Saved data video to {data_video_path}")

            # Save JSON with states and actions
            data_json_path = data_jsons_folder / f"{eval_idx}.json"
            json_data = {
                "state": state_list,
                "actions": action_list,
                "success": episode_success,
                "steps": t,
                "env_seed": env_seed,
                "video_path": str(data_video_path.absolute())
            }
            with open(data_json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
            print(f"Saved data JSON to {data_json_path}")

    # Print final statistics
    success_rate = (successes / total_episodes) * 100 if total_episodes > 0 else 0

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Environment: {env_id}")
    print(f"Total episodes: {total_episodes}")
    print(f"Successes: {successes}")
    print(f"Failures: {total_episodes - successes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print("="*80)

    # Print per-episode results
    print("\nPer-episode results:")
    for result in episode_results:
        status = "SUCCESS" if result['success'] else "FAILURE"
        print(f"  Episode {result['episode'] + 1}: {status} ({result['steps']} steps)")
    print("="*80)

    # Save results to file
    results_file = pathlib.Path(args.video_out_path) / "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Environment: {env_id}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Successes: {successes}\n")
        f.write(f"Failures: {total_episodes - successes}\n")
        f.write(f"Success Rate: {success_rate:.2f}%\n")
        f.write("="*80 + "\n\n")
        f.write("Per-episode results:\n")
        for result in episode_results:
            status = "SUCCESS" if result['success'] else "FAILURE"
            f.write(f"  Episode {result['episode'] + 1}: {status} ({result['steps']} steps)\n")

    logging.info(f"Results saved to {results_file}")

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
