#!/usr/bin/env python3
"""
VR teleop + policy inference + HDF5 collection (dataset_gen format).
Keys (terminal):
  s = start episode
  t = switch to teleop (label policy segment as 0)
  p = switch back to policy
  o = gripper open (toggle)
  c = gripper close (toggle)
  q = quit
"""
import os
import time
import sys
import threading
import queue
import termios
import tty
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import torch
import dill
import hydra
import h5py
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from tf2_msgs.msg import TFMessage
import spatialmath.base as smb
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R

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


class _TerminalKeyReader:
    def __init__(self):
        self._enabled = sys.stdin.isatty()
        self._queue = queue.Queue()
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _reader(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        while not self._stop.is_set():
            ch = sys.stdin.read(1)
            if ch:
                self._queue.put(ch)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_key(self):
        if not self._enabled:
            return None
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class _VRTrackerBridge(Node):
    def __init__(self, target_frame):
        super().__init__("vr_tracker_bridge")
        self.target_frame = target_frame
        self.T_station2track = None
        self._lock = threading.Lock()
        self.create_subscription(TFMessage, "/tf", self._tf_callback, 10)

    def _tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == self.target_frame:
                trans = transform.transform.translation
                rot = transform.transform.rotation
                pos = np.array([trans.x, trans.y, trans.z], dtype=np.float64)
                quat = [rot.w, rot.x, rot.y, rot.z]
                Rm = smb.q2r(quat)
                T = SE3.Rt(Rm, pos)
                with self._lock:
                    self.T_station2track = T

    def get_pose(self):
        with self._lock:
            if self.T_station2track is None:
                return None
            return self.T_station2track.copy()


def _start_vr_bridge(target_frame):
    rclpy.init(args=None)
    node = _VRTrackerBridge(target_frame)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return node, executor, thread


def _stop_vr_bridge(node, executor, thread):
    if executor is not None:
        executor.shutdown()
    if node is not None:
        node.destroy_node()
    rclpy.shutdown()
    if thread is not None:
        thread.join(timeout=1.0)


def _init_demo_buffer():
    return {
        "observations": {
            "joint": [],
            "image_wrist": [],
            "image_scene": [],
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
    img0 = _to_uint8(obs["image0"][-1])
    img1 = _to_uint8(obs["image1"][-1])
    joint = _get_joint_rad()
    gripper = float(obs["gripper"][-1][0]) if obs["gripper"].ndim > 1 else float(obs["gripper"][-1])

    buffer["observations"]["image_wrist"].append(img0)
    buffer["observations"]["image_scene"].append(img1)
    buffer["observations"]["joint"].append(joint)
    buffer["observations"]["gripper"].append(gripper)
    buffer["observations"]["advantage_indicator"].append(int(indicator_value))


def _set_indices_labels(buffer, indices, value):
    labels = buffer["observations"]["advantage_indicator"]
    for i in indices:
        labels[i] = int(value)


def _finalize_pending_labels(buffer, default_value=1):
    labels = buffer["observations"]["advantage_indicator"]
    for i in range(len(labels)):
        if labels[i] < 0:
            labels[i] = int(default_value)


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


def _wait_for_key(key_char, key_reader, idle_sleep=0.05):
    print(f"Press '{key_char}' in terminal to continue.")
    while True:
        key = key_reader.get_key()
        if key == key_char:
            return
        time.sleep(idle_sleep)


def _move_home(joints_deg, speed, acc):
    if joints_deg is None:
        return
    MoveJ(*[float(x) for x in joints_deg], float(speed), float(acc))
    time.sleep(2.0)


def _rotvec_to_rot6d(rotvec):
    Rm = R.from_rotvec(rotvec).as_matrix()
    return Rm[:, :2].reshape(-1)


def _se3_from_pos_quat(pos, quat):
    Rm = R.from_quat(quat).as_matrix()
    return SE3.Rt(Rm, np.array(pos))


def _rotation_safety_lock(delta_rot_rad, threshold_deg=178.0):
    max_rad = np.deg2rad(threshold_deg)
    min_rad = -max_rad
    margin = np.deg2rad(1.0)
    delta_rot_out = delta_rot_rad.copy()
    angle_lock = np.array([[False, False, False], [False, False, False]])
    locked_value = np.zeros(3)

    for i in range(3):
        raw = delta_rot_rad[i]

        if angle_lock[0][i]:
            if max_rad - margin > raw > 0:
                angle_lock[0][i] = False
            else:
                delta_rot_out[i] = locked_value[i]
                continue
        elif angle_lock[1][i]:
            if min_rad + margin < raw < 0:
                angle_lock[1][i] = False
            else:
                delta_rot_out[i] = locked_value[i]
                continue

        if raw >= max_rad:
            angle_lock[0][i] = True
            locked_value[i] = max_rad
            delta_rot_out[i] = max_rad
        elif raw <= min_rad:
            angle_lock[1][i] = True
            locked_value[i] = min_rad
            delta_rot_out[i] = min_rad

    return delta_rot_out


def _compute_vr_target_pose(
    T_station2track,
    T_tracker_init,
    init_pose,
    T_base2station,
    kp_pos,
    kp_rot,
    workspace_limits,
):
    T_rel = T_tracker_init.inv() * T_station2track
    delta_vr_pos = T_rel.t
    delta_vr_rot = smb.tr2rpy(T_rel.R, unit="rad")
    delta_vr_rot = _rotation_safety_lock(delta_vr_rot)

    R_base2station = T_base2station.R
    R_tracker_init = T_tracker_init.R
    R_end2base_init = init_pose.inv().R

    delta_pos_end = R_end2base_init @ R_base2station @ R_tracker_init @ delta_vr_pos
    delta_rot_end = R_end2base_init @ R_base2station @ R_tracker_init @ delta_vr_rot

    scaled_delta_pos = kp_pos * delta_pos_end
    scaled_delta_rot = kp_rot * np.array([delta_rot_end[0], delta_rot_end[1], delta_rot_end[2]])

    target_pose = init_pose * SE3(scaled_delta_pos) * SE3.RPY(*scaled_delta_rot, unit="rad")

    # clip workspace
    xlim, ylim, zlim = workspace_limits
    t = target_pose.t.copy()
    t[0] = np.clip(t[0], xlim[0], xlim[1])
    t[1] = np.clip(t[1], ylim[0], ylim[1])
    t[2] = np.clip(t[2], zlim[0], zlim[1])
    target_pose = SE3.Rt(target_pose.R, t)
    return target_pose


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
@click.option("--target_frame", default="vive_tracker", help="VR tracker frame name")
@click.option(
    "--home_joints",
    default="88.82315826,1.57005262,-108.45492554,16.88487434,-89.99609375,1.24207485",
    help="Home joint angles in deg, comma-separated (6 values)",
)
@click.option("--home_speed", default=60.0, type=float)
@click.option("--home_acc", default=80.0, type=float)
@click.option("--gripper_open", default=2100, type=int)
@click.option("--gripper_close", default=0, type=int)
@click.option("--kp_pos", default=0.9, type=float)
@click.option("--kp_rot", default=0.8, type=float)
def main(
    input,
    output,
    hdf5,
    robot_ip,
    vis_camera_idx,
    steps_per_inference,
    max_duration,
    frequency,
    command_latency,
    target_frame,
    home_joints,
    home_speed,
    home_acc,
    gripper_open,
    gripper_close,
    kp_pos,
    kp_rot,
):
    # parse home joints
    if home_joints:
        home_joints = np.array([float(x) for x in home_joints.split(",")], dtype=np.float32)
        if home_joints.shape[0] != 6:
            raise RuntimeError("home_joints must have 6 comma-separated values")
    else:
        home_joints = None

    # default calibration from servo_vr_ros2.py
    T_base2station = SE3.CopyFrom(
        np.array(
            [
                [-0.53399835, -0.01778211, 0.8452985, 6.52416805],
                [0.84533918, 0.00737158, 0.53417912, -3.26265108],
                [-0.01573002, 0.99981471, 0.0110955, 1.5960356],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        check=False,
    )

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

    key_reader = _TerminalKeyReader()
    key_reader.start()

    vr_node, vr_exec, vr_thread = _start_vr_bridge(target_frame)

    try:
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
                    _wait_for_key("s", key_reader)

                    demo_buffer = _init_demo_buffer()
                    policy_segment_indices = []
                    t_start = time.monotonic()
                    gripper_target = gripper_open
                    mode = "policy"
                    last_mode = "policy"

                    # initialize teleop reference
                    init_pose = None
                    tracker_init = None

                    while True:
                        key = key_reader.get_key()
                        if key == "q":
                            print("Quit requested.")
                            return
                        if key == "o":
                            gripper_target = gripper_open
                        if key == "c":
                            gripper_target = gripper_close
                        if key == "t":
                            if mode != "teleop":
                                _set_indices_labels(demo_buffer, policy_segment_indices, 0)
                                policy_segment_indices = []
                                mode = "teleop"
                                init_pose = None
                                tracker_init = None
                                print("Switched to teleop.")
                        if key == "p":
                            if mode != "policy":
                                _set_indices_labels(demo_buffer, policy_segment_indices, 1)
                                policy_segment_indices = []
                                mode = "policy"
                                print("Switched to policy.")

                        obs = env.get_obs()
                        frame = _to_uint8(obs[vis_rgb_key][-1])
                        cv2.imshow("rb10_vr_collect", frame[..., ::-1])
                        cv2.waitKey(1)

                        if mode == "policy":
                            # record current observation with pending label
                            _append_demo_obs(demo_buffer, obs, -1)
                            policy_segment_indices.append(len(demo_buffer["observations"]["advantage_indicator"]) - 1)

                            # policy inference + action execution
                            with torch.no_grad():
                                obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                result = policy.predict_action(obs_dict)
                                action = result["action"][0].detach().to("cpu").numpy()

                            # execute actions
                            this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                            this_target_poses[:, : action.shape[-1]] = action

                            action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset) * dt + obs[
                                "timestamp"
                            ][-1]
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

                            precise_wait(1.0 / frequency)

                        else:
                            # teleop mode: label as 1
                            _append_demo_obs(demo_buffer, obs, 1)

                            state = env.get_robot_state()
                            pos = state.get("robot_eef_pos", None)
                            quat = state.get("robot_eef_quat", None)
                            if pos is None or quat is None:
                                precise_wait(1.0 / frequency)
                                continue

                            if init_pose is None:
                                init_pose = _se3_from_pos_quat(pos, quat)

                            T_station2track = vr_node.get_pose()
                            if T_station2track is None:
                                precise_wait(1.0 / frequency)
                                continue

                            if tracker_init is None:
                                tracker_init = T_station2track

                            target_pose = _compute_vr_target_pose(
                                T_station2track,
                                tracker_init,
                                init_pose,
                                T_base2station,
                                kp_pos,
                                kp_rot,
                                workspace_limits=((-0.50, 0.50), (-0.90, -0.37), (0.095, 0.81)),
                            )

                            pos_t = target_pose.t
                            rotvec = R.from_matrix(target_pose.R).as_rotvec()
                            rot6d = _rotvec_to_rot6d(rotvec)
                            action = np.concatenate([pos_t, rot6d, [float(gripper_target)]])

                            t_cycle_end = time.monotonic() + dt
                            t_command_target = t_cycle_end + dt
                            env.exec_actions(
                                actions=[action],
                                timestamps=[t_command_target - time.monotonic() + time.time()],
                            )
                            precise_wait(t_cycle_end)

                        if time.monotonic() - t_start > max_duration:
                            print("Timeout reached.")
                            break

                    _finalize_pending_labels(demo_buffer, default_value=1)
                    if _prompt_yes_no("Save demo?"):
                        _save_demo_hdf5(demo_buffer, hdf5_path)

    finally:
        key_reader.stop()
        _stop_vr_bridge(vr_node, vr_exec, vr_thread)


if __name__ == "__main__":
    main()
