import os
import time
import pathlib
import copy
import h5py
import numpy as np
import cv2
import click
import torch
import dill
import hydra
from multiprocessing.managers import SharedMemoryManager

from diffusion_policy.real_world.rb10_real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.rb10_api.cobot import GetCurrentJoint, MoveJ


def _init_demo_buffer():
    return {
        "observations": {
            "image_wrist": [],
            "image_scene": [],
            "joint": [],
            "gripper": [],
            "advantage_indicator": [],
        }
    }


def _get_joint_rad():
    j = GetCurrentJoint()
    return np.array([j.j0, j.j1, j.j2, j.j3, j.j4, j.j5], dtype=np.float32) * np.pi / 180.0


def _to_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _append_demo_obs(buffer, obs, indicator_value):
    # image0/image1 are RGB from RealEnv; keep as uint8 for HDF5.
    img0 = _to_uint8(obs["image0"][-1])
    img1 = _to_uint8(obs["image1"][-1])

    joint = _get_joint_rad()
    gripper = float(obs["gripper"][-1][0]) if obs["gripper"].ndim > 1 else float(obs["gripper"][-1])

    buffer["observations"]["image_wrist"].append(img0)
    buffer["observations"]["image_scene"].append(img1)
    buffer["observations"]["joint"].append(joint)
    buffer["observations"]["gripper"].append(gripper)
    buffer["observations"]["advantage_indicator"].append(int(indicator_value))


def _set_pending_labels(buffer, value):
    labels = buffer["observations"]["advantage_indicator"]
    for i in range(len(labels)):
        if labels[i] < 0:
            labels[i] = int(value)


def _save_demo_hdf5(buffer, hdf5_path):
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "a") as f:
        if "data" not in f:
            data = f.create_group("data")
        else:
            data = f["data"]
        demo_idx = len(data.keys())
        demo = data.create_group(f"demo_{demo_idx}")
        obs_group = demo.create_group("observations")
        for key, values in buffer["observations"].items():
            obs_group.create_dataset(key, data=np.array(values))
    print(f"Saved demo_{demo_idx} to {hdf5_path}")


def _prompt_yes_no(msg):
    while True:
        ans = input(f"{msg} (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"


def _wait_for_key(key_char, idle_sleep=0.05):
    print(f"Press '{key_char}' to continue.")
    while True:
        key = cv2.pollKey()
        if key == ord(key_char):
            return
        time.sleep(idle_sleep)


def _move_home(joints, speed, acc):
    if joints is None:
        return
    MoveJ(joints, speed, acc)
    time.sleep(2.0)


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--output", "-o", required=True, help="Directory to save logs")
@click.option("--hdf5", default=None, help="Path to HDF5 file for saving demos")
@click.option("--robot_ip", "-ri", default="192.168.111.50", required=True, help="Robot IP")
@click.option("--vis_camera_idx", default=0, type=int, help="Which camera to visualize")
@click.option("--steps_per_inference", "-si", default=6, type=int, help="Action horizon for inference")
@click.option("--max_duration", "-md", default=60, help="Max duration per demo (sec)")
@click.option("--frequency", "-f", default=10, type=float, help="Control frequency in Hz")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="Command latency in sec")
@click.option("--home_joints", default=None, help="Home joint angles in rad, comma-separated (6 values)")
@click.option("--home_speed", default=1.05, type=float)
@click.option("--home_acc", default=1.4, type=float)
def main(input, output, hdf5, robot_ip, vis_camera_idx, steps_per_inference,
         max_duration, frequency, command_latency, home_joints, home_speed, home_acc):

    # parse home joints
    if home_joints:
        home_joints = np.array([float(x) for x in home_joints.split(",")], dtype=np.float32)
        if home_joints.shape[0] != 6:
            raise RuntimeError("home_joints must have 6 comma-separated values")
    else:
        home_joints = None

    # load checkpoint
    payload = torch.load(open(input, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    action_offset = 0
    delta_action = False
    if "diffusion" in cfg.name:
        policy: BaseImagePolicy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device("cuda")
        policy.eval().to(device)
        policy.num_inference_steps = 16
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    dt = 1 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps

    rgb_keys = [k for k, v in cfg.task.shape_meta["obs"].items() if v.get("type") == "rgb"]
    if len(rgb_keys) == 0:
        raise RuntimeError("No RGB keys found in shape_meta")
    if vis_camera_idx >= len(rgb_keys):
        raise RuntimeError("vis_camera_idx out of range for RGB keys")
    vis_rgb_key = rgb_keys[vis_camera_idx]

    hdf5_path = hdf5 or os.path.join(output, "inference_demos.hdf5")

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=False,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(1)
            print("Waiting for realsense...")
            time.sleep(1.0)

            # warm up
            obs = env.get_obs()
            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            _ = policy.predict_action(obs_dict)

            while True:
                _move_home(home_joints, home_speed, home_acc)
                _wait_for_key("s")

                demo_buffer = _init_demo_buffer()
                pending_label = True
                t_start = time.monotonic()
                iter_idx = 0

                while True:
                    obs = env.get_obs()
                    frame = _to_uint8(obs[vis_rgb_key][-1])
                    cv2.imshow("rb10_infer", frame[..., ::-1])
                    cv2.pollKey()

                    # record current observation
                    if pending_label:
                        _append_demo_obs(demo_buffer, obs, -1)
                    else:
                        _append_demo_obs(demo_buffer, obs, 0)

                    # inference + action execution
                    with torch.no_grad():
                        obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                        result = policy.predict_action(obs_dict)
                        action = result["action"][0].detach().to("cpu").numpy()

                    this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                    this_target_poses[:, :action.shape[-1]] = action

                    action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset) * dt + obs["timestamp"][-1]
                    curr_time = time.time()
                    is_new = action_timestamps > (curr_time + command_latency)
                    if np.sum(is_new) == 0:
                        this_target_poses = this_target_poses[[-1]]
                        next_step_idx = int(np.ceil((curr_time - time.time()) / dt))
                        action_timestamp = time.time() + (next_step_idx) * dt
                        action_timestamps = np.array([action_timestamp])
                    else:
                        this_target_poses = this_target_poses[is_new]
                        action_timestamps = action_timestamps[is_new]

                    env.exec_actions(actions=this_target_poses, timestamps=action_timestamps)

                    key = cv2.pollKey()
                    if key == ord("q"):
                        print("Quit requested.")
                        return
                    if key == ord("0"):
                        _set_pending_labels(demo_buffer, 0)
                        pending_label = False
                        if _prompt_yes_no("Start teleop now?"):
                            print("Teleop recording... Press 'c' to finish.")
                            while True:
                                obs = env.get_obs()
                                frame = _to_uint8(obs[vis_rgb_key][-1])
                                cv2.imshow("rb10_infer", frame[..., ::-1])
                                cv2.pollKey()
                                _append_demo_obs(demo_buffer, obs, 1)
                                key2 = cv2.pollKey()
                                if key2 == ord("c"):
                                    if _prompt_yes_no("Save demo?"):
                                        _save_demo_hdf5(demo_buffer, hdf5_path)
                                    break
                                if key2 == ord("q"):
                                    print("Quit requested.")
                                    return
                                precise_wait(1.0 / frequency)
                            break
                    if key == ord("1"):
                        _set_pending_labels(demo_buffer, 1)
                        pending_label = False
                        if _prompt_yes_no("Save demo?"):
                            _save_demo_hdf5(demo_buffer, hdf5_path)
                        break

                    if time.monotonic() - t_start > max_duration:
                        print("Timeout reached.")
                        break

                    precise_wait(1.0 / frequency)


if __name__ == "__main__":
    main()
