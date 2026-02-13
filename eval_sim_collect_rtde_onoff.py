import os
import pathlib
import threading
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

DEFAULT_INPUT_CKPT = "/home/jy/Downloads/idp_sim.ckpt"
DEFAULT_LOG_DIR = "/home/jy/Desktop/folk_idp_sim/log"
DEFAULT_HDF5_PATH = "/home/jy/Desktop/folk_idp_sim/inference_data/sim_inference_demos.hdf5"


class ViveROSControl:
    def __init__(self, topic):
        self.lock = threading.Lock()
        self.topic = topic
        self.latest_pose = None
        self.prev_pose = None
        self.last_update = 0.0
        self.count = 0
        self.base_pos = None

        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        self._context = Context()
        self._context.init(args=None)
        self._executor = SingleThreadedExecutor(context=self._context)

        class _ViveNode(Node):
            def __init__(self, parent, context):
                super().__init__("vive_teleop_listener", context=context)
                self.parent = parent
                self.create_subscription(Odometry, parent.topic, self.cb, 10)

            def cb(self, msg):
                pos = np.array(
                    [
                        msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z,
                    ],
                    dtype=np.float32,
                )
                quat = np.array(
                    [
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                    ],
                    dtype=np.float32,
                )
                with self.parent.lock:
                    self.parent.latest_pose = (pos, quat)
                    self.parent.last_update = time.time()
                    self.parent.count += 1

        self._node = _ViveNode(self, self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def set_base(self):
        with self.lock:
            if self.latest_pose is None:
                self.base_pos = None
            else:
                self.base_pos = self.latest_pose[0].copy()
                self.prev_pose = None

    def get_delta(self):
        with self.lock:
            if self.latest_pose is None:
                return None, None, self.last_update
            pos, quat = self.latest_pose
            if self.base_pos is not None:
                pos = pos - self.base_pos
            if self.prev_pose is None:
                self.prev_pose = (pos.copy(), quat.copy())
                return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), self.last_update
            prev_pos, prev_quat = self.prev_pose
            delta = pos - prev_pos
            r_prev = R.from_quat(prev_quat)
            r_curr = R.from_quat(quat)
            delta_r = (r_curr * r_prev.inv()).as_rotvec().astype(np.float32)
            self.prev_pose = (pos.copy(), quat.copy())
            return delta.astype(np.float32), delta_r, self.last_update

    def close(self):
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            self._context.shutdown()
        except Exception:
            pass
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass


def _map_vive_axes(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec.astype(np.float32)
    return np.array([-z, -x, y], dtype=np.float32)


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


def _show_collect_images(obs_win: dict, status_text: str, window_name: str = "sim_collect"):
    def to_bgr(img_chw: np.ndarray) -> np.ndarray:
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        img_hwc = np.clip(img_hwc, 0.0, 1.0)
        img_u8 = (img_hwc * 255.0).astype(np.uint8)
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    wrist = to_bgr(obs_win["image0"][-1])
    full = to_bgr(obs_win["image1"][-1])
    # Keep simulation data unchanged and flip only the displayed second view vertically.
    full = cv2.flip(full, 0)
    panel = np.concatenate([wrist, full], axis=1)
    cv2.putText(panel, "image0: wrist", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "image1: full (inference input)",
        (wrist.shape[1] + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(panel, status_text, (12, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(window_name, panel)


def _render_bgr(sim, height, width, camera_name=None):
    if camera_name is None:
        img = sim.render(height=height, width=width)
    else:
        img = sim.render(height=height, width=width, camera_name=camera_name)
    img = img[..., ::-1]
    img = np.flip(img, axis=0)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return np.ascontiguousarray(img)


def _label_for_camera(name):
    if name in ("eye_in_hand", "robot0_eye_in_hand"):
        return "1st-person"
    if name in ("frontview", "agentview", "robotview", "robot0_robotview"):
        return "front"
    if name in ("backview",):
        return "back"
    return name or "view"


def _pick_front_camera(sim):
    try:
        names = list(sim.model.camera_names)
    except Exception:
        return None
    for name in ("robot0_eye_in_hand", "eye_in_hand"):
        if name in names:
            return name
    for name in ("frontview", "agentview", "robotview", "robot0_robotview", "sideview", "birdview"):
        if name in names:
            return name
    return names[0] if names else None


def _pick_back_camera(sim):
    try:
        names = list(sim.model.camera_names)
    except Exception:
        return None
    if "backview" in names:
        return "backview"
    if "frontview" in names:
        return "frontview"
    return names[0] if names else None


def _show_data_collection_view(sim, front_camera_name, back_camera_name, status_text, window_name="sim_collect"):
    sim_width = 640
    sim_height = 360

    sim_img = _render_bgr(sim, sim_height, sim_width, camera_name=front_camera_name)
    h, w = sim_img.shape[:2]
    origin = (80, h - 80)
    axis_len = 60
    cv2.arrowedLine(sim_img, origin, (origin[0] + axis_len, origin[1]), (0, 0, 255), 3, tipLength=0.2)
    cv2.putText(sim_img, "X", (origin[0] + axis_len + 8, origin[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.arrowedLine(sim_img, origin, (origin[0], origin[1] - axis_len), (0, 255, 0), 3, tipLength=0.2)
    cv2.putText(sim_img, "Y", (origin[0] - 18, origin[1] - axis_len - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.arrowedLine(sim_img, origin, (origin[0] - axis_len // 2, origin[1] + axis_len // 2), (255, 0, 0), 3, tipLength=0.2)
    cv2.putText(
        sim_img,
        "Z",
        (origin[0] - axis_len // 2 - 18, origin[1] + axis_len // 2 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    back_img = _render_bgr(sim, sim_height, sim_width, camera_name=back_camera_name)
    front_label = f"{_label_for_camera(front_camera_name)} ({front_camera_name})"
    back_label = f"{_label_for_camera(back_camera_name)} ({back_camera_name})"
    cv2.putText(sim_img, front_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(back_img, back_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(back_img, status_text, (20, sim_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    split_img = np.hstack([sim_img, back_img])
    cv2.imshow(window_name, split_img)


def _show_message_panel(lines, window_name="sim_collect", width=960, height=240):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = 60
    for line in lines:
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        y += 40
    cv2.imshow(window_name, panel)


def _mark_recent_policy_segment_bad(indicators, step_modes, bad_label=0):
    """
    Mark the most recent contiguous policy segment as bad.
    This is called when user switches from policy -> teleop intervention.
    """
    i = len(step_modes) - 1
    while i >= 0 and step_modes[i] == "policy":
        indicators[i] = bad_label
        i -= 1


@click.command()
@click.option("--input", "-i", default=DEFAULT_INPUT_CKPT, show_default=True, help="Path to checkpoint")
@click.option("--output", "-o", default=DEFAULT_LOG_DIR, show_default=True, help="Directory to save non-HDF5 logs")
@click.option("--hdf5", default=DEFAULT_HDF5_PATH, show_default=True, help="Path to output HDF5")
@click.option("--task", default="son_pick_and_place_image_adv", show_default=True)
@click.option("--max_steps", default=1000, type=int, show_default=True)
@click.option("--frequency", default=10.0, type=float, show_default=True)
@click.option("--steps_per_inference", default=6, type=int, show_default=True)
@click.option("--advantage_inference_value", default=1, type=int, show_default=True)
@click.option("--policy_label", default=0, type=int, show_default=True)
@click.option("--teleop_label", default=1, type=int, show_default=True)
@click.option("--osc_pos_output_max", default=0.05, type=float, show_default=True)
@click.option("--osc_rot_output_max", default=0.5, type=float, show_default=True)
@click.option("--binarize_gripper/--continuous_gripper", default=True, show_default=True)
@click.option("--vive_topic", default="/vive_tracker_ros/raw_pose", show_default=True, help="ROS2 Odometry topic from vive_tracker_ros2")
@click.option("--vive_pos_scale", default=65.0, type=float, show_default=True, help="Position gain from Vive delta to sim teleop pose")
@click.option("--vive_rot_scale", default=4.0, type=float, show_default=True, help="Rotation gain from Vive delta-rotvec to sim teleop pose")
@click.option("--vive_stale_timeout", default=0.5, type=float, show_default=True, help="Ignore Vive updates older than this many seconds")
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
    vive_topic,
    vive_pos_scale,
    vive_rot_scale,
    vive_stale_timeout,
):
    _ = task
    os.makedirs(output, exist_ok=True)
    hdf5_path = hdf5
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

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

    vive_control = ViveROSControl(vive_topic)

    print(f"[vive] subscribed topic: {vive_topic}")
    print("Controls: s=start episode, t=teleop(Vive), p=policy, v=reset Vive base, x=end episode, q=quit")
    print("Teleop gripper: open=left/o/[ , close=right/c/] (press g to swap open/close mapping)")
    print("Indicator rule: default=1, and when switching policy->teleop, recent policy segment is relabeled to 0")
    cv2.namedWindow("sim_collect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("sim_collect", 1280, 720)

    try:
        quit_all = False
        while True:
            _show_message_panel(
                [
                    "Waiting: press 's' to start next demo",
                    "Press 'q' to quit",
                ]
            )
            key = cv2.waitKeyEx(30)
            if key in (ord("q"), ord("Q")):
                break
            if key not in (ord("s"), ord("S")):
                continue

            obs = env.reset()
            sync = ObsWindowSync(n_obs_steps=cfg.n_obs_steps)
            sync.reset(obs)
            low, high = env.action_spec()
            front_camera_name = _pick_front_camera(env.env.sim)
            back_camera_name = _pick_back_camera(env.env.sim)

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
            done = False
            gripper_swap = False
            demo = {
                "obs": {"image0": [], "image1": [], "position": [], "quat": [], "gripper": [], "advantage_indicator": []},
                "actions": [],
                "step_mode": [],
            }

            while (not done) and (step_count < max_steps):
                obs_win = sync.get_window()
                _show_data_collection_view(
                    env.env.sim,
                    front_camera_name=front_camera_name,
                    back_camera_name=back_camera_name,
                    status_text=f"mode={mode} step={step_count}",
                )

                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q")):
                    quit_all = True
                    break
                if key in (ord("x"), ord("X")):
                    break
                if key in (ord("t"), ord("T")):
                    if mode == "policy":
                        # Policy segment right before intervention is considered bad.
                        _mark_recent_policy_segment_bad(
                            demo["obs"]["advantage_indicator"], demo["step_mode"], bad_label=0
                        )
                    mode = "teleop"
                    vive_control.set_base()
                elif key in (ord("p"), ord("P")):
                    mode = "policy"
                elif key in (ord("v"), ord("V")):
                    vive_control.set_base()
                    print("[vive] base reset")
                elif key in (ord("g"), ord("G")):
                    gripper_swap = not gripper_swap
                    print(f"[teleop] gripper mapping swapped={gripper_swap}")

                a_exec = None
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
                    label = 1
                else:
                    # In teleop mode, use direct OSC delta command (no interpolation).
                    a_exec = np.zeros_like(low, dtype=np.float32)
                    delta, delta_r, last_update = vive_control.get_delta()
                    if delta is not None and (time.time() - float(last_update)) <= float(vive_stale_timeout):
                        dpos = _map_vive_axes(delta) * float(vive_pos_scale)
                        drot = _map_vive_axes(delta_r) * float(vive_rot_scale)
                        a_exec[0] = dpos[0]
                        a_exec[1] = dpos[1]
                        a_exec[2] = dpos[2]
                        a_exec[3] = drot[0]
                        a_exec[4] = drot[1]
                        a_exec[5] = drot[2]
                    if a_exec.shape[0] >= 7:
                        open_cmd = float(low[6])
                        close_cmd = float(high[6])
                        if gripper_swap:
                            open_cmd, close_cmd = close_cmd, open_cmd
                        if key in (81, 2424832, 65361, ord("o"), ord("O"), ord("[")):  # left arrow / o / [
                            a_exec[6] = open_cmd
                        elif key in (83, 2555904, 65363, ord("c"), ord("C"), ord("]")):  # right arrow / c / ]
                            a_exec[6] = close_cmd
                    a_exec = np.clip(a_exec, low, high)
                    label = 1

                if mode == "policy":
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
                demo["step_mode"].append(mode)

                t_cycle_end = t_start + (step_count + steps_per_inference) * dt
                precise_wait(t_cycle_end, time_func=time.monotonic)
                step_count += 1

            if quit_all:
                break

            if len(demo["actions"]) > 0:
                while True:
                    _show_message_panel(
                        [
                            f"Episode finished: {len(demo['actions'])} steps",
                            "Save this demo? press 'y' (save) or 'n' (discard)",
                            "Press 'q' to quit now",
                        ]
                    )
                    key = cv2.waitKeyEx(30)
                    if key in (ord("y"), ord("Y")):
                        _save_demo_hdf5(demo, hdf5_path)
                        print(f"Episode saved: steps={len(demo['actions'])}")
                        break
                    if key in (ord("n"), ord("N")):
                        print("Episode discarded")
                        break
                    if key in (ord("q"), ord("Q")):
                        quit_all = True
                        break
            else:
                print("Episode skipped (no data)")

            env.reset()
            print("Returned to initial pose. Waiting for next demo (press 's' to start, 'q' to quit).")
            if quit_all:
                break

    finally:
        cv2.destroyAllWindows()
        try:
            vive_control.close()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
