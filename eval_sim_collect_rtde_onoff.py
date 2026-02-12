import os
import pathlib
import time

import click
import cv2
import dill
import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.sim_world import ObsWindowSync, SimEnvAdapterSonAdv
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_proj = np.dot(b1, a2) * b1
    b2 = a2 - a2_proj
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=1)


def _gripper_abs_to_cmd(gripper_abs: float, binarize: bool) -> float:
    g = float(gripper_abs)
    if 0.0 <= g <= 1.0:
        g = 2.0 * g - 1.0
    else:
        g = float(np.clip(g, -1.0, 1.0))
    if binarize:
        g = 1.0 if g >= 0.0 else -1.0
    return g


def _action_to_abs_pose7(action_raw: np.ndarray, obs_now: dict, pos_output_max: float, rot_output_max: float) -> np.ndarray:
    action_raw = np.asarray(action_raw, dtype=np.float32).reshape(-1)
    curr_pos = np.asarray(obs_now["position"], dtype=np.float32).reshape(3)
    curr_quat_xyzw = np.asarray(obs_now["quat"], dtype=np.float32).reshape(4)
    curr_rot = R.from_quat(curr_quat_xyzw).as_matrix()

    if action_raw.shape[0] == 10:
        target_pos = action_raw[:3]
        target_rot = _rot6d_to_matrix(action_raw[3:9])
        target_gripper_abs = float(action_raw[9])
    elif action_raw.shape[0] == 8:
        target_pos = action_raw[:3]
        target_quat_xyzw = action_raw[3:7]
        target_quat_xyzw = target_quat_xyzw / (np.linalg.norm(target_quat_xyzw) + 1e-8)
        target_rot = R.from_quat(target_quat_xyzw).as_matrix()
        target_gripper_abs = float(action_raw[7])
    elif action_raw.shape[0] == 7:
        dpos = action_raw[:3] * float(pos_output_max)
        drot_axis_angle = action_raw[3:6] * float(rot_output_max)
        target_pos = curr_pos + dpos
        target_rot = R.from_rotvec(drot_axis_angle).as_matrix() @ curr_rot
        g_cmd = float(np.clip(action_raw[6], -1.0, 1.0))
        target_gripper_abs = 0.5 * (g_cmd + 1.0)
    else:
        raise ValueError(f"Unsupported action dim {action_raw.shape[0]} for waypoint scheduling")

    target_rotvec = R.from_matrix(target_rot).as_rotvec().astype(np.float32)
    return np.concatenate([target_pos.astype(np.float32), target_rotvec, np.array([target_gripper_abs], dtype=np.float32)])


def _abs_pose7_to_osc_delta_action(
    target_pose7: np.ndarray,
    obs_now: dict,
    low: np.ndarray,
    high: np.ndarray,
    pos_output_max: float,
    rot_output_max: float,
    binarize_gripper: bool,
) -> np.ndarray:
    target_pose7 = np.asarray(target_pose7, dtype=np.float32).reshape(7)
    target_pos = target_pose7[:3]
    target_rot = R.from_rotvec(target_pose7[3:6]).as_matrix()
    target_gripper_abs = float(target_pose7[6])

    curr_pos = np.asarray(obs_now["position"], dtype=np.float32).reshape(3)
    curr_quat_xyzw = np.asarray(obs_now["quat"], dtype=np.float32).reshape(4)
    curr_rot = R.from_quat(curr_quat_xyzw).as_matrix()

    delta_rot = target_rot @ curr_rot.T
    drot_axis_angle = R.from_matrix(delta_rot).as_rotvec().astype(np.float32)
    dpos = (target_pos - curr_pos).astype(np.float32)

    pos_cmd = dpos / float(pos_output_max)
    rot_cmd = drot_axis_angle / float(rot_output_max)
    grip_cmd = np.array([_gripper_abs_to_cmd(target_gripper_abs, binarize_gripper)], dtype=np.float32)
    action7 = np.concatenate([pos_cmd, rot_cmd, grip_cmd], axis=0).astype(np.float32)
    return np.clip(action7, low, high)


def _obs_to_hdf5_item(obs_now: dict) -> dict:
    image0_hwc = (np.transpose(obs_now["image0"], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    image1_hwc = (np.transpose(obs_now["image1"], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    return {
        "image0": image0_hwc,
        "image1": image1_hwc,
        "position": np.asarray(obs_now["position"], dtype=np.float32),
        "quat": np.asarray(obs_now["quat"], dtype=np.float32),
        "gripper": np.asarray(obs_now["gripper"], dtype=np.float32),
    }


def _save_demo_hdf5(demo: dict, hdf5_path: str):
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "a") as f:
        data = f["data"] if "data" in f else f.create_group("data")
        demo_idx = len(data.keys())
        grp = data.create_group(f"demo_{demo_idx}")
        obs_grp = grp.create_group("obs")
        for k, v in demo["obs"].items():
            obs_grp.create_dataset(k, data=np.asarray(v))
        grp.create_dataset("actions", data=np.asarray(demo["actions"], dtype=np.float32))
    print(f"Saved demo_{demo_idx} -> {hdf5_path}")


def _draw_status(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(frame_bgr, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame_bgr


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--output", "-o", required=True, help="Directory to save logs")
@click.option("--hdf5", default=None, help="Path to output HDF5")
@click.option("--task", default="son_pick_and_place_image_adv", show_default=True)
@click.option("--max_steps", default=300, type=int, show_default=True)
@click.option("--frequency", default=10.0, type=float, show_default=True)
@click.option("--steps_per_inference", default=6, type=int, show_default=True)
@click.option("--advantage_inference_value", default=1, type=int, show_default=True)
@click.option("--policy_label", default=0, type=int, show_default=True)
@click.option("--teleop_label", default=1, type=int, show_default=True)
@click.option("--osc_pos_output_max", default=0.05, type=float, show_default=True)
@click.option("--osc_rot_output_max", default=0.5, type=float, show_default=True)
@click.option("--binarize_gripper/--continuous_gripper", default=True, show_default=True)
def main(
    input,
    output,
    hdf5,
    task,
    max_steps,
    frequency,
    steps_per_inference,
    advantage_inference_value,
    policy_label,
    teleop_label,
    osc_pos_output_max,
    osc_rot_output_max,
    binarize_gripper,
):
    _ = task
    os.makedirs(output, exist_ok=True)
    hdf5_path = hdf5 or os.path.join(output, "sim_inference_demos.hdf5")

    payload = torch.load(open(input, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    if hasattr(policy, "advantage_inference_value"):
        policy.advantage_inference_value = int(advantage_inference_value)
    policy.reset()

    dt = 1.0 / float(frequency)
    env = SimEnvAdapterSonAdv(
        env_name="Lift",
        robots="Panda",
        camera_names=("robot0_eye_in_hand", "frontview"),
        camera_heights=240,
        camera_widths=320,
        control_freq=int(max(1, round(frequency))),
        horizon=max_steps,
        has_renderer=False,
        has_offscreen_renderer=True,
    )

    print("Controls: s=start episode, t=teleop, p=policy, x=end/save, q=quit")
    print("Teleop keys: w/s x, a/d y, r/f z, o=open gripper, c=close gripper")

    try:
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key != ord("s"):
                continue

            obs = env.reset()
            sync = ObsWindowSync(n_obs_steps=cfg.n_obs_steps)
            sync.reset(obs)
            low, high = env.action_spec()

            curr_rotvec = R.from_quat(np.asarray(obs["quat"], dtype=np.float32).reshape(4)).as_rotvec().astype(np.float32)
            curr_gripper = float(np.asarray(obs["gripper"], dtype=np.float32).reshape(-1)[0])
            init_pose7 = np.concatenate(
                [np.asarray(obs["position"], dtype=np.float32).reshape(3), curr_rotvec, np.asarray([curr_gripper], dtype=np.float32)]
            )
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([time.monotonic()], dtype=np.float64),
                poses=np.array([init_pose7], dtype=np.float64),
            )
            last_waypoint_time = pose_interp.times[-1]
            mode = "policy"
            step_count = 0
            t_start = time.monotonic()
            teleop_pose = init_pose7.copy()
            done = False
            demo = {
                "obs": {"image0": [], "image1": [], "position": [], "quat": [], "gripper": [], "advantage_indicator": []},
                "actions": [],
            }

            while (not done) and (step_count < max_steps):
                obs_win = sync.get_window()
                frame = (np.transpose(obs_win["image1"][-1], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
                frame_bgr = _draw_status(frame[..., ::-1].copy(), f"mode={mode} step={step_count}")
                cv2.imshow("sim_collect", frame_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return
                if key == ord("x"):
                    break
                if key == ord("t"):
                    mode = "teleop"
                elif key == ord("p"):
                    mode = "policy"

                if mode == "policy":
                    with torch.no_grad():
                        obs_dict_np = {
                            "image0": obs_win["image0"],
                            "image1": obs_win["image1"],
                            "position": obs_win["position"],
                            "quat": obs_win["quat"],
                            "gripper": obs_win["gripper"],
                        }
                        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                        result = policy.predict_action(obs_dict)
                        action_seq = result["action"][0].detach().to("cpu").numpy()
                    raw_action = action_seq[0]
                    target_pose7 = _action_to_abs_pose7(raw_action, obs, osc_pos_output_max, osc_rot_output_max)
                    target_time_mono = time.monotonic() + dt
                    pose_interp = pose_interp.schedule_waypoint(
                        pose=target_pose7,
                        time=target_time_mono,
                        max_pos_speed=np.inf,
                        max_rot_speed=np.inf,
                        curr_time=time.monotonic(),
                        last_waypoint_time=last_waypoint_time,
                    )
                    last_waypoint_time = pose_interp.times[-1]
                    label = int(policy_label)
                else:
                    dpos = np.zeros(3, dtype=np.float32)
                    if key == ord("w"):
                        dpos[0] += 0.01
                    elif key == ord("s"):
                        dpos[0] -= 0.01
                    elif key == ord("a"):
                        dpos[1] += 0.01
                    elif key == ord("d"):
                        dpos[1] -= 0.01
                    elif key == ord("r"):
                        dpos[2] += 0.01
                    elif key == ord("f"):
                        dpos[2] -= 0.01
                    teleop_pose[:3] = teleop_pose[:3] + dpos
                    if key == ord("o"):
                        teleop_pose[6] = 1.0
                    elif key == ord("c"):
                        teleop_pose[6] = 0.0
                    pose_interp = pose_interp.schedule_waypoint(
                        pose=teleop_pose.copy(),
                        time=time.monotonic() + dt,
                        max_pos_speed=np.inf,
                        max_rot_speed=np.inf,
                        curr_time=time.monotonic(),
                        last_waypoint_time=last_waypoint_time,
                    )
                    last_waypoint_time = pose_interp.times[-1]
                    label = int(teleop_label)

                t_cmd = time.time() + dt
                precise_wait(t_cmd, time_func=time.time)
                interp_pose7 = pose_interp(time.monotonic())
                a_exec = _abs_pose7_to_osc_delta_action(
                    target_pose7=interp_pose7,
                    obs_now=obs,
                    low=low,
                    high=high,
                    pos_output_max=osc_pos_output_max,
                    rot_output_max=osc_rot_output_max,
                    binarize_gripper=binarize_gripper,
                )
                obs, _, done, _ = env.step(a_exec)
                sync.push(obs)

                item = _obs_to_hdf5_item(obs)
                demo["obs"]["image0"].append(item["image0"])
                demo["obs"]["image1"].append(item["image1"])
                demo["obs"]["position"].append(item["position"])
                demo["obs"]["quat"].append(item["quat"])
                demo["obs"]["gripper"].append(item["gripper"])
                demo["obs"]["advantage_indicator"].append(label)
                demo["actions"].append(a_exec.astype(np.float32))

                t_cycle_end = t_start + (step_count + steps_per_inference) * dt
                precise_wait(t_cycle_end, time_func=time.monotonic)
                step_count += 1

            if len(demo["actions"]) > 0:
                _save_demo_hdf5(demo, hdf5_path)
                print(f"Episode saved: steps={len(demo['actions'])}")
            else:
                print("Episode skipped (no data)")

    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
