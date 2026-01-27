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
  x = end episode (prompt save)
"""
import os
import math
from collections import deque
import time
import sys
import threading
import faulthandler
import signal
import queue
import termios
import tty
import subprocess
from multiprocessing.managers import BaseManager, SharedMemoryManager

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
from std_msgs.msg import UInt8, Int32
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R

from diffusion_policy.real_world.multi_realsense import MultiRealsense
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.rb10_api.cobot import MoveJ
from diffusion_policy.rb.RB10 import RB10


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


class _RobotStateNode(Node):
    def __init__(self, robot_model: RB10):
        super().__init__("rb10_state_node")
        self.robot_model = robot_model
        self._lock = threading.Lock()
        self._joint_buf = deque(maxlen=2000)
        self._gripper_buf = deque(maxlen=2000)
        self._teleop_pub = self.create_publisher(UInt8, "/teleop_control", 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)
        self.create_subscription(Int32, "/gripper/present_position", self._gripper_cb, 50)

    def _joint_cb(self, msg):
        if len(msg.position) < 6:
            return
        t = time.time()
        with self._lock:
            self._joint_buf.append((t, np.array(msg.position[:6], dtype=np.float64)))

    def _gripper_cb(self, msg):
        t = time.time()
        with self._lock:
            self._gripper_buf.append((t, float(msg.data)))

    def set_teleop(self, enabled: bool, wait_sub: bool = True, timeout: float = 2.0):
        msg = UInt8()
        msg.data = 1 if enabled else 0
        if wait_sub:
            t0 = time.time()
            while self._teleop_pub.get_subscription_count() == 0:
                if time.time() - t0 > timeout:
                    break
                time.sleep(0.05)
        # publish a few times to avoid missing the first message
        for _ in range(3):
            self._teleop_pub.publish(msg)
            time.sleep(0.05)

    def get_last_k(self, k: int):
        with self._lock:
            joints = list(self._joint_buf)
            grippers = list(self._gripper_buf)
        return joints, grippers


def _start_ros_node(node: Node):
    if not rclpy.ok():
        rclpy.init(args=None)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread


def _stop_ros_node(node: Node, executor, thread):
    if executor is not None:
        executor.shutdown()
    if node is not None:
        node.destroy_node()
    if rclpy.ok():
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


def _get_joint_rad(joint_rad: np.ndarray):
    return np.array(joint_rad, dtype=np.float32)


def _to_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _append_demo_obs(buffer, obs, indicator_value):
    img0 = _to_uint8(obs["image0"][-1])
    img1 = _to_uint8(obs["image1"][-1])
    if "joint" in obs:
        joint = _get_joint_rad(obs["joint"][-1])
    else:
        joint = np.zeros((6,), dtype=np.float32)
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


def _select_indices_by_timestamp(timestamps, target_times):
    idxs = []
    for t in target_times:
        is_before = np.nonzero(timestamps < t)[0]
        if len(is_before) > 0:
            idxs.append(is_before[-1])
        else:
            idxs.append(0)
    return idxs


def _robot_obs_from_rtde(rtde_ctrl, target_times, n_obs_steps):
    robot_k = max(n_obs_steps * 3, 10)
    last_robot_data = rtde_ctrl.get_state(k=robot_k)
    robot_timestamps = last_robot_data["robot_receive_timestamp"]
    if len(robot_timestamps) == 0:
        return None
    idxs = _select_indices_by_timestamp(robot_timestamps, target_times)

    pos = last_robot_data["robot_eef_pos"][idxs]
    quat = last_robot_data["robot_eef_quat"][idxs]
    gripper = last_robot_data["robot_gripper_qpos"][idxs]
    return {
        "position": pos,
        "quat": quat,
        "gripper": gripper,
        "timestamp": target_times,
    }


def _robot_obs_from_ros(joint_buf, gripper_buf, robot_model, target_times):
    if len(joint_buf) == 0:
        return None
    joint_times = np.array([x[0] for x in joint_buf], dtype=np.float64)
    joints = np.stack([x[1] for x in joint_buf], axis=0)
    if len(gripper_buf) > 0:
        gripper_times = np.array([x[0] for x in gripper_buf], dtype=np.float64)
        grippers = np.array([x[1] for x in gripper_buf], dtype=np.float64)
    else:
        gripper_times = joint_times
        grippers = np.zeros((len(joint_times),), dtype=np.float64)

    joint_idxs = _select_indices_by_timestamp(joint_times, target_times)
    grip_idxs = _select_indices_by_timestamp(gripper_times, target_times)

    pos = []
    quat = []
    for q in joints[joint_idxs]:
        T = robot_model.fkine(q)
        pos.append(T.t)
        quat.append(R.from_matrix(T.R).as_quat())
    pos = np.stack(pos, axis=0)
    quat = np.stack(quat, axis=0)
    gripper = grippers[grip_idxs].reshape(-1, 1)

    return {
        "position": pos,
        "quat": quat,
        "gripper": gripper,
        "joint": joints[joint_idxs],
        "timestamp": target_times,
    }


def _camera_obs(realsense, last_realsense_data, n_obs_steps, frequency, capture_fps):
    k = math.ceil(n_obs_steps * (capture_fps / frequency))
    last_realsense_data = realsense.get(k=k, out=last_realsense_data)
    camera_last_timestamp = np.max([x["timestamp"][-1] for x in last_realsense_data.values()])
    dt = 1.0 / frequency
    obs_times = camera_last_timestamp - (np.arange(n_obs_steps)[::-1] * dt)
    camera_obs = {}
    for camera_idx, value in last_realsense_data.items():
        this_timestamps = value["timestamp"]
        idxs = _select_indices_by_timestamp(this_timestamps, obs_times)
        camera_obs[f"image{camera_idx}"] = value["color"][idxs]
    return camera_obs, obs_times, last_realsense_data


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


def _prompt_yes_no(msg, key_reader, idle_sleep=0.05):
    # Flush stale keys.
    while True:
        stale = key_reader.get_key()
        if stale is None:
            break
    print(f"{msg} (y/n): ", end="", flush=True)
    while True:
        key = key_reader.get_key()
        if key in ("y", "n", "Y", "N", "1", "0"):
            print(key)
            return key in ("y", "Y", "1")
        time.sleep(idle_sleep)


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




@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--output", "-o", required=True, help="Directory to save logs")
@click.option("--hdf5", default=None, help="Path to HDF5 file for saving demos")
@click.option("--robot_ip", "-ri", default="192.168.111.50", required=True, help="Robot IP")
@click.option("--vis_camera_idx", default=0, type=int, help="Which camera to visualize")
@click.option("--steps_per_inference", "-si", default=6, type=int, help="Action horizon for inference")
@click.option("--max_duration", "-md", default=100, help="Max duration per demo (sec)")
@click.option("--frequency", "-f", default=20, type=float, help="Control frequency in Hz")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="Command latency in sec")
@click.option("--target_frame", default="vive_tracker", help="VR tracker frame name")
@click.option("--rtde_port", default=50051, type=int, help="RTDE proxy server port")
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
    rtde_port,
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
    realsense = None
    state_node = None
    ros_exec = None
    ros_thread = None
    rtde_proc = None
    rtde_proxy = None
    rtde_mgr = None
    servo_proc = None

    servo_cmd = (
        "bash -lc 'source /opt/ros/humble/setup.bash && "
        "source ~/rb_ws/install/setup.bash && "
        "python3 /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts/servo_vr_ros2_on_demand.py "
        f"--robot_ip {robot_ip} --target_frame {target_frame}'"
    )
    rtde_authkey = "rb10_rtde"
    rtde_cmd = None

    class _RTDEManager(BaseManager):
        pass

    _RTDEManager.register("RTDEProxy")

    def _start_servo():
        nonlocal servo_proc
        if servo_proc is not None and servo_proc.poll() is None:
            return
        servo_proc = subprocess.Popen(servo_cmd, shell=True)
        print(f"[SERVO] started (pid={servo_proc.pid})", flush=True)

    def _stop_servo():
        nonlocal servo_proc
        if servo_proc is None:
            return
        if servo_proc.poll() is None:
            servo_proc.terminate()
        servo_proc = None

    # teleop_control is published via ROS node

    try:
        # Debug: dump stack traces if initialization hangs.
        faulthandler.enable()
        faulthandler.dump_traceback_later(30, repeat=True)
        def _sigusr1_handler(signum, frame):
            faulthandler.dump_traceback()
        signal.signal(signal.SIGUSR1, _sigusr1_handler)

        # Clean up stale processes from previous runs
        try:
            subprocess.run("pkill -f rtde_proxy_server.py", shell=True, check=False)
            subprocess.run("pkill -f servo_vr_ros2_on_demand.py", shell=True, check=False)
        except Exception:
            pass

        print("[DEBUG] before SharedMemoryManager", flush=True)
        with SharedMemoryManager() as shm_manager:
            print("[DEBUG] after SharedMemoryManager", flush=True)

            video_capture_fps = 30
            video_capture_resolution = (640, 480)
            max_obs_buffer_size = 30
            rtde_cmd = (
                "bash -lc '"
                "source /opt/ros/humble/setup.bash && "
                "python3 /home/son_rb/interactive_diffusion_policy/rtde_proxy_server.py "
                f"--robot_ip {robot_ip} --port {rtde_port} --authkey {rtde_authkey} "
                "--frequency 125 --lookahead_time 0.1 --gain 300 "
                "--max_pos_speed 0.25 --max_rot_speed 0.6 "
                f"--launch_timeout 5 --get_max_k {max_obs_buffer_size}"
                "'"
            )

            color_tf = get_image_transform(
                input_res=video_capture_resolution,
                output_res=obs_res,
                bgr_to_rgb=True
            )

            def transform(data):
                data["color"] = color_tf(data["color"]).astype(np.float32) / 255.0
                return data

            realsense = MultiRealsense(
                serial_numbers=None,
                shm_manager=shm_manager,
                resolution=video_capture_resolution,
                capture_fps=video_capture_fps,
                put_fps=video_capture_fps,
                put_downsample=False,
                get_max_k=max_obs_buffer_size,
                transform=transform,
                vis_transform=None,
                recording_transform=None,
                video_recorder=None,
                verbose=False
            )
            realsense.start(wait=False)
            realsense.start_wait()

            robot_model = RB10()

            key_reader.start()
            last_realsense_data = None
            rtde_proc = None
            rtde_proxy = None
            rtde_mgr = None

            def _start_rtde():
                nonlocal rtde_proc, rtde_proxy, rtde_mgr
                # Try connect to existing proxy first
                if rtde_proxy is not None:
                    try:
                        rtde_proxy.ping()
                        ok = rtde_proxy.start_rtde()
                        if ok:
                            print("[RTDE] ready", flush=True)
                            return
                    except Exception:
                        rtde_proxy = None
                        rtde_mgr = None
                # Kill any stale proxy on the same port
                try:
                    subprocess.run("pkill -f rtde_proxy_server.py", shell=True, check=False)
                except Exception:
                    pass
                print("[RTDE] starting RTDE process...", flush=True)
                if rtde_proc is None or rtde_proc.poll() is not None:
                    rtde_proc = subprocess.Popen(rtde_cmd, shell=True)
                t0 = time.time()
                while True:
                    try:
                        rtde_mgr = _RTDEManager(
                            address=("127.0.0.1", rtde_port),
                            authkey=rtde_authkey.encode("utf-8"),
                        )
                        rtde_mgr.connect()
                        rtde_proxy = rtde_mgr.RTDEProxy()
                        rtde_proxy.ping()
                        ok = rtde_proxy.start_rtde()
                        if not ok:
                            raise RuntimeError("RTDE proxy failed to start RTDE controller.")
                        print("[RTDE] ready", flush=True)
                        return
                    except Exception:
                        if rtde_proc.poll() is not None:
                            raise RuntimeError("RTDE proxy process exited early.")
                        if time.time() - t0 > 6.0:
                            raise RuntimeError("RTDE proxy not ready after 6s.")
                        time.sleep(0.1)

            def _stop_rtde_controller():
                nonlocal rtde_proc, rtde_proxy, rtde_mgr
                try:
                    if rtde_proxy is not None:
                        rtde_proxy.stop_rtde()
                except Exception:
                    pass
                # keep proxy process alive to avoid port conflicts

            def _stop_rtde_process():
                nonlocal rtde_proc, rtde_proxy, rtde_mgr
                try:
                    if rtde_proxy is not None:
                        rtde_proxy.stop_rtde()
                except Exception:
                    pass
                rtde_proxy = None
                rtde_mgr = None
                if rtde_proc is not None and rtde_proc.poll() is None:
                    rtde_proc.terminate()
                rtde_proc = None

            print("Waiting for realsense...")
            time.sleep(1.0)

            if not rclpy.ok():
                rclpy.init(args=None)
            state_node = _RobotStateNode(robot_model)
            ros_exec, ros_thread = _start_ros_node(state_node)

            # warm up (camera + rtde state)
            print("[DEBUG] warmup: before get_obs", flush=True)
            _start_rtde()
            cam_obs, obs_times, last_realsense_data = _camera_obs(
                realsense, last_realsense_data, n_obs_steps, frequency, video_capture_fps
            )
            robot_obs = _robot_obs_from_rtde(rtde_proxy, obs_times, n_obs_steps)
            if robot_obs is None:
                raise RuntimeError("No RTDE robot state received for warmup.")
            joints, grippers = state_node.get_last_k(n_obs_steps)
            if len(joints) > 0:
                joint_obs = _robot_obs_from_ros(joints, grippers, robot_model, obs_times)
                if joint_obs is not None and "joint" in joint_obs:
                    robot_obs["joint"] = joint_obs["joint"]
            obs = dict(cam_obs)
            obs.update(robot_obs)
            obs["timestamp"] = obs_times
            print("[DEBUG] warmup: after get_obs", flush=True)
            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            _ = policy.predict_action(obs_dict)
            print("[DEBUG] warmup: after predict_action", flush=True)
            faulthandler.cancel_dump_traceback_later()

            while True:
                _wait_for_key("s", key_reader)
                print("Controls: t=teleop, p=policy, o=open, c=close, x=end, q=quit", flush=True)

                demo_buffer = _init_demo_buffer()
                policy_segment_indices = []
                t_start = time.monotonic()
                gripper_target = gripper_open
                mode = "policy"

                _stop_servo()
                _start_rtde()
                state_node.set_teleop(False)

                while True:
                    key = key_reader.get_key()
                    if key == "q":
                        print("Quit requested.")
                        return
                    if key == "x":
                        print("Episode end requested.")
                        break
                    if key == "o":
                        gripper_target = gripper_open
                    if key == "c":
                        gripper_target = gripper_close
                    if key == "t":
                        if mode != "teleop":
                            _stop_rtde_controller()
                            _start_servo()
                            state_node.set_teleop(True)
                            _set_indices_labels(demo_buffer, policy_segment_indices, 0)
                            policy_segment_indices = []
                            mode = "teleop"
                            print("Switched to teleop.")
                    if key == "p":
                        if mode != "policy":
                            state_node.set_teleop(False)
                            _stop_servo()
                            _start_rtde()
                            _set_indices_labels(demo_buffer, policy_segment_indices, 1)
                            policy_segment_indices = []
                            mode = "policy"
                            print("Switched to policy.")

                    cam_obs, obs_times, last_realsense_data = _camera_obs(
                        realsense, last_realsense_data, n_obs_steps, frequency, video_capture_fps
                    )
                    if vis_rgb_key in cam_obs:
                        frame = _to_uint8(cam_obs[vis_rgb_key][-1])
                        cv2.imshow("rb10_vr_collect", frame[..., ::-1])
                        cv2.waitKey(1)
                    if mode == "policy":
                        robot_obs = _robot_obs_from_rtde(rtde_proxy, obs_times, n_obs_steps)
                    else:
                        joints, grippers = state_node.get_last_k(n_obs_steps)
                        robot_obs = _robot_obs_from_ros(joints, grippers, robot_model, obs_times)
                    if robot_obs is None:
                        precise_wait(1.0 / frequency)
                        continue
                    if "joint" not in robot_obs:
                        joints, grippers = state_node.get_last_k(n_obs_steps)
                        joint_obs = _robot_obs_from_ros(joints, grippers, robot_model, obs_times)
                        if joint_obs is not None and "joint" in joint_obs:
                            robot_obs["joint"] = joint_obs["joint"]

                    obs = dict(cam_obs)
                    obs.update(robot_obs)
                    obs["timestamp"] = obs_times

                    if mode == "policy":
                        _append_demo_obs(demo_buffer, obs, -1)
                        policy_segment_indices.append(len(demo_buffer["observations"]["advantage_indicator"]) - 1)

                        with torch.no_grad():
                            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            action = result["action"][0].detach().to("cpu").numpy()

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

                        if rtde_proxy is not None:
                            for i in range(len(this_target_poses)):
                                rtde_proxy.schedule_waypoint(
                                    pose=this_target_poses[i],
                                    target_time=float(action_timestamps[i])
                                )
                        precise_wait(1.0 / frequency)
                    else:
                        _append_demo_obs(demo_buffer, obs, 1)
                        precise_wait(1.0 / frequency)

                    if time.monotonic() - t_start > max_duration:
                        print("Timeout reached.")
                        break

                state_node.set_teleop(False)
                _stop_rtde_process()
                _stop_servo()

                _finalize_pending_labels(demo_buffer, default_value=1)
                if _prompt_yes_no("Save demo?", key_reader):
                    _save_demo_hdf5(demo_buffer, hdf5_path)

    finally:
        key_reader.stop()
        if state_node is not None:
            state_node.set_teleop(False)
        if rtde_proxy is not None or rtde_proc is not None:
            try:
                _stop_rtde_process()
            except Exception:
                pass
        if servo_proc is not None:
            try:
                if servo_proc.poll() is None:
                    servo_proc.terminate()
            except Exception:
                pass
        if realsense is not None:
            try:
                realsense.stop(wait=False)
                realsense.stop_wait()
            except Exception:
                pass
        _stop_ros_node(state_node, ros_exec, ros_thread)


if __name__ == "__main__":
    main()
