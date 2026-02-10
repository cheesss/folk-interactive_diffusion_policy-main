import json
import os
import pathlib
import time

import click
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.sim_world import (
    SimEnvAdapterSonAdv,
    ObsWindowSync,
    ActionScheduler,
    EvalMetrics,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _show_inference_images(obs_win: dict, window_name: str = "inference_input"):
    import cv2

    def to_bgr(img_chw: np.ndarray) -> np.ndarray:
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        img_hwc = np.clip(img_hwc, 0.0, 1.0)
        img_u8 = (img_hwc * 255.0).astype(np.uint8)
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    wrist = to_bgr(obs_win["image0"][-1])
    full = to_bgr(obs_win["image1"][-1])
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
    cv2.imshow(window_name, panel)
    cv2.waitKey(1)


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


def _to_osc_delta_action(
    action_raw: np.ndarray,
    obs_now: dict,
    low: np.ndarray,
    high: np.ndarray,
    pos_output_max: float,
    rot_output_max: float,
    binarize_gripper: bool,
) -> np.ndarray:
    action_raw = np.asarray(action_raw, dtype=np.float32).reshape(-1)
    if action_raw.shape[0] == low.shape[0]:
        return np.clip(action_raw, low, high)

    if action_raw.shape[0] not in (8, 10):
        raise ValueError(
            f"Unsupported action dim {action_raw.shape[0]}. "
            f"Expected env action dim {low.shape[0]} or absolute 8D (xyz+quat+gripper) / 10D (xyz+rot6d+gripper)."
        )
    if low.shape[0] != 7:
        raise ValueError(
            f"Absolute 10D conversion currently targets OSC 7D env action. "
            f"Current env action dim: {low.shape[0]}"
        )

    target_pos = action_raw[:3]
    if action_raw.shape[0] == 10:
        target_rot = _rot6d_to_matrix(action_raw[3:9])
        target_gripper_abs = action_raw[9]
    else:
        target_quat_xyzw = action_raw[3:7]
        target_quat_xyzw = target_quat_xyzw / (np.linalg.norm(target_quat_xyzw) + 1e-8)
        target_rot = R.from_quat(target_quat_xyzw).as_matrix()
        target_gripper_abs = action_raw[7]

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


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--output", "-o", required=True, help="Directory to save logs")
@click.option("--task", default="son_pick_and_place_image_adv", show_default=True)
@click.option("--n_episodes", default=1, type=int, show_default=True)
@click.option("--max_steps", default=200, type=int, show_default=True)
@click.option("--frequency", default=10.0, type=float, show_default=True)
@click.option("--steps_per_inference", default=6, type=int, show_default=True)
@click.option("--num_inference_steps", default=None, type=int)
@click.option("--advantage_inference_value", default=1, type=int, show_default=True)
@click.option("--osc_pos_output_max", default=0.05, type=float, show_default=True)
@click.option("--osc_rot_output_max", default=0.5, type=float, show_default=True)
@click.option("--binarize_gripper/--continuous_gripper", default=True, show_default=True)
@click.option("--render", is_flag=True, default=False)
@click.option("--show_inference_images", is_flag=True, default=False)
@click.option("--print_gripper", is_flag=True, default=False)
def main(
    input,
    output,
    task,
    n_episodes,
    max_steps,
    frequency,
    steps_per_inference,
    num_inference_steps,
    advantage_inference_value,
    osc_pos_output_max,
    osc_rot_output_max,
    binarize_gripper,
    render,
    show_inference_images,
    print_gripper,
):
    _ = task  # reserved for compatibility with task-specific launch style
    os.makedirs(output, exist_ok=True)

    payload = torch.load(open(input, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if "diffusion" not in cfg.name:
        raise RuntimeError(f"Unsupported policy type: {cfg.name}")

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)

    if num_inference_steps is not None:
        policy.num_inference_steps = num_inference_steps

    if hasattr(policy, "advantage_inference_value"):
        policy.advantage_inference_value = int(advantage_inference_value)

    dt = 1.0 / float(frequency)
    scheduler = ActionScheduler(dt=dt, action_offset=0, exec_latency=0.01)

    env = SimEnvAdapterSonAdv(
        env_name="Lift",
        robots="Panda",
        camera_names=("robot0_eye_in_hand", "frontview"),
        camera_heights=240,
        camera_widths=320,
        control_freq=int(max(1, round(frequency))),
        horizon=max_steps,
        has_renderer=render,
        # Image observations require offscreen rendering even when on-screen render is enabled.
        has_offscreen_renderer=True,
    )

    metrics_all = []

    try:
        for ep in range(n_episodes):
            obs = env.reset()
            sync = ObsWindowSync(n_obs_steps=cfg.n_obs_steps)
            sync.reset(obs)
            metrics = EvalMetrics()

            low, high = env.action_spec()
            done = False

            for _ in range(max_steps):
                if render:
                    env.render()

                obs_win = sync.get_window()
                if show_inference_images:
                    _show_inference_images(obs_win)
                obs_dict_np = {
                    "image0": obs_win["image0"],
                    "image1": obs_win["image1"],
                    "position": obs_win["position"],
                    "quat": obs_win["quat"],
                    "gripper": obs_win["gripper"],
                }

                t0 = time.time()
                with torch.no_grad():
                    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action(obs_dict)
                    action_seq = result["action"][0].detach().to("cpu").numpy()
                infer_ms = (time.time() - t0) * 1000.0
                metrics.add_infer_ms(infer_ms)

                kept_actions, dropped, kept = scheduler.schedule(action_seq, time.time())
                cycle_actions = scheduler.take_cycle(kept_actions, steps_per_inference)
                metrics.add_action_counts(len(action_seq), len(cycle_actions), dropped + (kept - len(cycle_actions)))

                if len(cycle_actions) == 0:
                    cycle_actions = action_seq[[-1]]

                for a in cycle_actions:
                    obs_gripper_in = float(np.asarray(obs["gripper"]).reshape(-1)[0])
                    raw_gripper = float(np.asarray(a).reshape(-1)[-1])
                    a_exec = _to_osc_delta_action(
                        action_raw=a,
                        obs_now=obs,
                        low=low,
                        high=high,
                        pos_output_max=osc_pos_output_max,
                        rot_output_max=osc_rot_output_max,
                        binarize_gripper=binarize_gripper,
                    )
                    if print_gripper:
                        exec_gripper = float(np.asarray(a_exec).reshape(-1)[-1])
                        print(
                            f"[gripper] obs_in={obs_gripper_in:.4f} "
                            f"raw_out={raw_gripper:.4f} exec_out={exec_gripper:.4f}"
                        )
                    obs, r, done, _ = env.step(a_exec)
                    metrics.add_reward(float(r))
                    sync.push(obs)
                    if done:
                        break
                if done:
                    break

            summary = metrics.summary()
            summary["episode"] = ep
            metrics_all.append(summary)
            print(json.dumps(summary, indent=2))

        out_path = pathlib.Path(output).joinpath("eval_sim_scheduled_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, indent=2)
        print(f"Saved metrics to {out_path}")
    finally:
        if show_inference_images:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        env.close()


if __name__ == "__main__":
    main()
