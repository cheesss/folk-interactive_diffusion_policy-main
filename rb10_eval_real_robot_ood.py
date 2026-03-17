#!/home/vision/anaconda3/envs/robodiff/bin/python

"""
원본 rb10_eval_real_robot.py를 건드리지 않고,
실시간 OOD 모니터링과 시각화를 분리한 엔트리포인트.
"""

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
import pathlib
import imageio.v2 as imageio
from omegaconf import OmegaConf

from diffusion_policy.real_world.ood_monitor import OODMonitor
from diffusion_policy.real_world.rb10_real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

import rb10_eval_real_robot as base_eval


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _get_last_step_obs(obs_dict):
    return dict_apply(obs_dict, lambda x: x[:, -1])


def _validate_obs_dict_np(obs_dict_np, shape_meta):
    expected_obs = shape_meta["obs"]
    actual_keys = set(obs_dict_np.keys())
    expected_keys = set(expected_obs.keys())

    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(f"Observation key mismatch. missing={missing}, extra={extra}")

    for key, attr in expected_obs.items():
        expected_type = attr.get("type", "low_dim")
        expected_shape = tuple(attr["shape"])
        value = obs_dict_np[key]

        if value.ndim != len(expected_shape) + 1:
            raise ValueError(
                f"Observation rank mismatch for '{key}': "
                f"expected {len(expected_shape) + 1} dims (T + feature), got {value.ndim} with shape {tuple(value.shape)}"
            )

        actual_feature_shape = tuple(value.shape[1:])
        if actual_feature_shape != expected_shape:
            raise ValueError(
                f"Observation shape mismatch for '{key}': "
                f"expected per-step shape {expected_shape}, got {actual_feature_shape}"
            )

        if expected_type == "rgb":
            if value.dtype not in (np.float32, np.float64, np.uint8):
                raise ValueError(f"Unexpected dtype for rgb key '{key}': {value.dtype}")
        elif expected_type == "low_dim":
            if not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"Unexpected dtype for low_dim key '{key}': {value.dtype}")
        else:
            raise RuntimeError(f"Unsupported obs type in shape_meta: {expected_type}")


def _draw_metric_plot(values, width, height, title, color, threshold=None):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin = 32
    x0, y0 = margin, margin
    x1, y1 = width - margin, height - margin
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (30, 30, 30), 1)

    y_max = 1.2
    if len(values) > 0:
        y_max = max(y_max, float(np.max(values)) * 1.15)
    if threshold is not None:
        y_max = max(y_max, threshold * 1.2)
    y_max = max(y_max, 1e-6)

    tick_values = np.linspace(0.0, y_max, num=5)
    for tick in tick_values:
        y_tick = int(y1 - (tick / y_max) * (y1 - y0))
        cv2.line(canvas, (x0, y_tick), (x1, y_tick), (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{tick:.1f}",
            (4, y_tick + 4),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.42,
            color=(80, 80, 80),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    if len(values) >= 2:
        xs = np.linspace(x0, x1, num=len(values))
        ys = []
        for value in values:
            t = np.clip(value / y_max, 0.0, 1.0)
            ys.append(y1 - t * (y1 - y0))
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[-1]), 4, color, -1, cv2.LINE_AA)

    if threshold is not None:
        threshold_y = y1 - np.clip(threshold / y_max, 0.0, 1.0) * (y1 - y0)
        cv2.line(canvas, (x0, int(threshold_y)), (x1, int(threshold_y)), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"thr={threshold:.2f}",
            (x0 + 6, int(threshold_y) - 6),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.45,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    latest = values[-1] if len(values) > 0 else 0.0
    cv2.putText(
        canvas,
        f"{title}: {latest:.2f}",
        (x0, y0 - 6),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "old",
        (x0, y1 + 18),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.42,
        color=(80, 80, 80),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "now",
        (x1 - 28, y1 + 18),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.42,
        color=(80, 80, 80),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return canvas


def _render_ood_panel(
    frame_bgr,
    current_series,
    predicted_series,
    current_score,
    predicted_score,
    threshold,
    mode="policy",
):
    frame = frame_bgr.copy()
    border_color = (0, 0, 255) if current_score >= threshold else (40, 160, 40)
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), border_color, 4)

    panel_width = max(320, frame.shape[1] // 2)
    header_height = 96
    body_height = frame.shape[0] - header_height
    half_body = body_height // 2

    header = np.full((header_height, panel_width, 3), 248, dtype=np.uint8)
    curr_alert = "ALERT" if current_score >= threshold else "OK"
    pred_text = f"{predicted_score:.2f}" if predicted_score is not None else "N/A"
    lines = [
        f"mode: {mode}",
        f"current_ood: {current_score:.2f} ({curr_alert})",
        f"predicted_ood: {pred_text}",
        f"threshold: {threshold:.2f}",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            header,
            line,
            (14, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.58,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        y += 21

    current_plot = _draw_metric_plot(
        current_series,
        panel_width,
        half_body,
        title="current_ood",
        color=(0, 120, 255),
        threshold=threshold,
    )
    predicted_plot = _draw_metric_plot(
        predicted_series if len(predicted_series) > 0 else [0.0],
        panel_width,
        frame.shape[0] - header_height - half_body,
        title="predicted_ood",
        color=(50, 50, 220),
        threshold=threshold,
    )
    panel = np.vstack([header, current_plot, predicted_plot])
    return np.concatenate([frame, panel], axis=1)


def _write_ood_video(out_path, frames_bgr, fps):
    out_path = str(out_path)
    try:
        with imageio.get_writer(out_path, fps=fps, format="FFMPEG") as writer:
            for frame_bgr in frames_bgr:
                writer.append_data(frame_bgr[..., ::-1])
        return out_path
    except Exception:
        fallback = out_path + ".gif" if not out_path.lower().endswith(".gif") else out_path
        with imageio.get_writer(fallback, fps=fps) as writer:
            for frame_bgr in frames_bgr:
                writer.append_data(frame_bgr[..., ::-1])
        return fallback


def _save_ood_video_if_needed(output_dir, episode_idx, frames_bgr, fps):
    if len(frames_bgr) == 0:
        return None
    out_dir = pathlib.Path(output_dir).joinpath("ood_videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath(f"episode_{episode_idx:04d}.mp4")
    return _write_ood_video(out_path, frames_bgr, fps)


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option("--robot_ip", "-ri", default="192.168.111.50", required=True, help="UR5 IP address")
@click.option("--match_dataset", "-m", default=None, help="Dataset used to overlay and adjust initial condition")
@click.option("--match_episode", "-me", default=None, type=int, help="Match specific episode from the match dataset")
@click.option("--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize.")
@click.option("--init_joints", "-j", is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option("--steps_per_inference", "-si", default=6, type=int, help="Action horizon for inference.")
@click.option("--max_duration", "-md", default=60, help="Max duration for each epoch in seconds.")
@click.option("--frequency", "-f", default=10, type=float, help="Control frequency in Hz.")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="Latency between receiving command and executing on robot in seconds.")
@click.option("--valuef", default=None, help="Value model checkpoint path for live graph")
@click.option("--value_vis_every", default=3, type=int, help="Save value frames every N loops")
@click.option("--oodf", default=None, help="OOD dynamics checkpoint path")
@click.option("--ood_bank", default=None, help="Expert latent bank path")
@click.option("--ood_threshold", default=None, type=float, help="Raw OOD threshold override")
@click.option("--ood_vis_every", default=1, type=int, help="Save OOD frames every N loops")
@click.option("--ood_plot_window", default=120, type=int, help="Number of recent OOD points to plot")
@click.option("--ood_chunk_size", default=4096, type=int, help="Chunk size for kNN OOD distance")
@click.option("--debug_timing", is_flag=True, default=False, help="Print timing markers for debugging")
def main(
    input,
    output,
    robot_ip,
    match_dataset,
    match_episode,
    vis_camera_idx,
    init_joints,
    steps_per_inference,
    max_duration,
    frequency,
    command_latency,
    valuef,
    value_vis_every,
    oodf,
    ood_bank,
    ood_threshold,
    ood_vis_every,
    ood_plot_window,
    ood_chunk_size,
    debug_timing,
):
    ckpt_path = input
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
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

    value_cfg = None
    value_model = None
    if valuef is not None:
        value_cfg, value_model = base_eval._load_value_model(valuef, device)

    ood_monitor = None
    if ood_bank is not None:
        ood_monitor = OODMonitor(
            policy=policy,
            bank_path=ood_bank,
            expected_shape_meta=cfg.task.shape_meta,
            dynamics_ckpt=oodf,
            threshold=ood_threshold,
            device=str(device),
            chunk_size=ood_chunk_size,
        )

    dt = 1 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)
    rgb_keys = base_eval._get_rgb_keys(cfg.task.shape_meta)
    if len(rgb_keys) == 0:
        raise RuntimeError("No RGB keys found in shape_meta")
    if vis_camera_idx >= len(rgb_keys):
        raise RuntimeError("vis_camera_idx out of range for RGB keys")
    vis_rgb_key = rgb_keys[vis_camera_idx]
    episode_idx = 0

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(1)
            key_reader = base_eval._TerminalKeyReader()
            key_reader.start()

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                _validate_obs_dict_np(obs_dict_np, cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result["action"][0].detach().to("cpu").numpy()
                assert action.shape[-1] == 10
                if ood_monitor is not None:
                    last_obs = _get_last_step_obs(obs_dict)
                    _ = ood_monitor.score_current(last_obs)
                    _ = ood_monitor.score_predicted(last_obs, result["action"])
                del result

            print("Ready!")
            stop_all = False
            while True:
                value_frames = []
                value_series = []
                value_frame_indices = []
                ood_frames = []
                ood_current_series = []
                ood_pred_series = []
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay

                    env.start_episode(eval_t_start)
                    episode_idx += 1
                    frame_latency = 1 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    paused = False
                    while True:
                        key = key_reader.get_key()
                        if key == "s":
                            if not paused:
                                paused = True
                                print("Paused. Press 'c' to continue, 'q' to quit.")
                        elif key == "c":
                            if paused:
                                paused = False
                                eval_t_start = time.time()
                                t_start = time.monotonic()
                                iter_idx = 0
                                print("Resumed.")
                        elif key == "q" or key == "\x03":
                            stop_all = True
                            print("Stopping and exiting.")
                            env.end_episode()
                            if value_model is not None:
                                saved = base_eval._save_value_video_if_needed(
                                    output, episode_idx, value_frames, value_series, value_frame_indices, fps=frequency
                                )
                                if saved is not None:
                                    print(f"Saved value video to {saved}")
                            if ood_monitor is not None:
                                saved = _save_ood_video_if_needed(output, episode_idx, ood_frames, fps=frequency)
                                if saved is not None:
                                    print(f"Saved OOD video to {saved}")
                            break

                        if paused:
                            time.sleep(0.05)
                            continue

                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        if debug_timing:
                            print(f"[TIMING] before_get_obs t={time.time():.3f}")
                        print("get_obs")
                        obs = env.get_obs()
                        if debug_timing:
                            print(f"[TIMING] after_get_obs t={time.time():.3f}")
                        obs_timestamps = obs["timestamp"]
                        print(f"Obs latency {time.time() - obs_timestamps[-1]}")

                        with torch.no_grad():
                            if debug_timing:
                                print(f"[TIMING] before_policy_infer t={time.time():.3f}")
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                            _validate_obs_dict_np(obs_dict_np, cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            action = result["action"][0].detach().to("cpu").numpy()
                            print("Inference latency:", time.time() - s)
                            if debug_timing:
                                print(f"[TIMING] after_policy_infer t={time.time():.3f}")

                            if value_model is not None:
                                if debug_timing:
                                    print(f"[TIMING] before_value_infer t={time.time():.3f}")
                                value_obs_np = get_real_obs_dict(env_obs=obs, shape_meta=value_cfg.shape_meta)
                                value_obs = dict_apply(value_obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                logits = value_model(value_obs)
                                pred_bin = int(torch.argmax(logits, dim=-1).item())
                                value_series.append(pred_bin)
                                if debug_timing:
                                    print(f"[TIMING] after_value_infer t={time.time():.3f}")

                                frame = base_eval._normalize_vis_image(obs[vis_rgb_key][-1]).copy()
                                if (iter_idx % max(1, value_vis_every)) == 0:
                                    value_frames.append(frame)
                                    value_frame_indices.append(len(value_series) - 1)

                            if ood_monitor is not None:
                                last_obs = _get_last_step_obs(obs_dict)
                                current_metrics = ood_monitor.score_current(last_obs)
                                predicted_metrics = ood_monitor.score_predicted(last_obs, result["action"])

                                ood_current_series.append(current_metrics["normalized"])
                                pred_score = None
                                if predicted_metrics is not None:
                                    pred_score = predicted_metrics["normalized"]
                                    ood_pred_series.append(pred_score)

                                vis_frame = _render_ood_panel(
                                    frame_bgr=base_eval._normalize_vis_image(obs[vis_rgb_key][-1])[..., ::-1].copy(),
                                    current_series=ood_current_series[-ood_plot_window:],
                                    predicted_series=ood_pred_series[-ood_plot_window:],
                                    current_score=current_metrics["normalized"],
                                    predicted_score=pred_score,
                                    threshold=ood_monitor.threshold_normalized,
                                    mode="policy",
                                )
                                cv2.imshow("rb10_ood", vis_frame)
                                cv2.waitKey(1)
                                if (iter_idx % max(1, ood_vis_every)) == 0:
                                    ood_frames.append(vis_frame.copy())
                            else:
                                cv2.imshow("rb10_eval", base_eval._normalize_vis_image(obs[vis_rgb_key][-1])[..., ::-1])
                                cv2.waitKey(1)

                        if delta_action:
                            assert len(action) == 1
                            if perv_target_pose is None:
                                perv_target_pose = obs["robot_eef_pose"][-1]
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose[[0, 1]] += action[-1]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:
                            this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                            this_target_poses[:, : action.shape[-1]] = action

                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)

                        if np.sum(is_new) == 0:
                            this_target_poses = this_target_poses[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + next_step_idx * dt
                            print("Over budget", action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        if debug_timing:
                            print(f"[TIMING] before_exec_actions t={time.time():.3f}")
                        env.exec_actions(actions=this_target_poses, timestamps=action_timestamps)
                        if debug_timing:
                            print(f"[TIMING] after_exec_actions t={time.time():.3f}")
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print("Terminated by the timeout!")

                        if terminate:
                            env.end_episode()
                            if value_model is not None:
                                saved = base_eval._save_value_video_if_needed(
                                    output, episode_idx, value_frames, value_series, value_frame_indices, fps=frequency
                                )
                                if saved is not None:
                                    print(f"Saved value video to {saved}")
                            if ood_monitor is not None:
                                saved = _save_ood_video_if_needed(output, episode_idx, ood_frames, fps=frequency)
                                if saved is not None:
                                    print(f"Saved OOD video to {saved}")
                            break

                        if debug_timing:
                            print(f"[TIMING] before_wait t={time.time():.3f}")
                        precise_wait(t_cycle_end - frame_latency)
                        if debug_timing:
                            print(f"[TIMING] after_wait t={time.time():.3f}")
                        iter_idx += steps_per_inference
                    if stop_all:
                        break

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()
                    if value_model is not None:
                        saved = base_eval._save_value_video_if_needed(
                            output, episode_idx, value_frames, value_series, value_frame_indices, fps=frequency
                        )
                        if saved is not None:
                            print(f"Saved value video to {saved}")
                    if ood_monitor is not None:
                        saved = _save_ood_video_if_needed(output, episode_idx, ood_frames, fps=frequency)
                        if saved is not None:
                            print(f"Saved OOD video to {saved}")

                print("Stopped.")
                if stop_all:
                    break
            key_reader.stop()


if __name__ == "__main__":
    main()
