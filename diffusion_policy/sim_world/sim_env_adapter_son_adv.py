import os
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class SimEnvAdapterSonAdv:
    """Adapter that maps robosuite Lift observations to SON ADV policy keys."""

    def __init__(
        self,
        env_name: str = "Lift",
        robots: str = "Panda",
        camera_names=("robot0_eye_in_hand", "frontview"),
        camera_heights=240,
        camera_widths=320,
        control_freq=10,
        horizon=400,
        has_renderer=False,
        has_offscreen_renderer=True,
    ):
        # On Windows, MuJoCo/robosuite does not accept Linux-style "egl".
        if os.name == "nt" and os.environ.get("MUJOCO_GL", "").lower() == "egl":
            os.environ["MUJOCO_GL"] = "glfw"

        import robosuite as suite
        controller_config = None
        try:
            # robosuite <= 1.4 style
            from robosuite.controllers import load_controller_config

            controller_config = load_controller_config(default_controller="OSC_POSE")
        except Exception:
            # robosuite >= 1.5 style
            from robosuite.controllers import load_composite_controller_config

            controller_config = load_composite_controller_config(
                controller="BASIC",
                robot=robots,
            )
            # Keep OSC pose behavior for single-arm robots when possible.
            if isinstance(controller_config, dict):
                if "body_parts" in controller_config and "arms" in controller_config["body_parts"]:
                    controller_config["body_parts"]["arms"]["type"] = "OSC_POSE"
                elif "arm" in controller_config:
                    controller_config["arm"]["type"] = "OSC_POSE"

        # Prefer local custom environment implementation in fork_idp/simenv.py.
        custom_env_cls = None
        simenv_path = Path(__file__).resolve().parents[3] / "simenv.py"
        if simenv_path.exists():
            spec = importlib.util.spec_from_file_location("fork_simenv", str(simenv_path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                custom_env_cls = getattr(mod, env_name, None) or getattr(mod, "Lift", None)

        if custom_env_cls is not None:
            self.env = custom_env_cls(
                robots=robots,
                controller_configs=controller_config,
                use_camera_obs=True,
                has_renderer=has_renderer,
                has_offscreen_renderer=has_offscreen_renderer,
                camera_names=list(camera_names),
                camera_heights=camera_heights,
                camera_widths=camera_widths,
                control_freq=control_freq,
                horizon=horizon,
            )
        else:
            self.env = suite.make(
                env_name=env_name,
                robots=robots,
                has_renderer=has_renderer,
                has_offscreen_renderer=has_offscreen_renderer,
                use_camera_obs=True,
                camera_names=list(camera_names),
                camera_heights=camera_heights,
                camera_widths=camera_widths,
                controller_configs=controller_config,
                control_freq=control_freq,
                horizon=horizon,
            )

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        return self._convert_obs(obs)

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        return self._convert_obs(obs), float(reward), bool(done), info

    def close(self):
        self.env.close()

    def render(self):
        try:
            self.env.render()
        except Exception:
            pass

    def action_spec(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.env.action_spec

    def _convert_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        # robosuite returns camera frames in HWC uint8; policy expects CHW float in [0, 1].
        out["image0"] = np.transpose(obs["robot0_eye_in_hand_image"], (2, 0, 1)).astype(np.float32) / 255.0
        out["image1"] = np.transpose(obs["frontview_image"], (2, 0, 1)).astype(np.float32) / 255.0
        out["position"] = obs["robot0_eef_pos"].astype(np.float32)
        out["quat"] = obs["robot0_eef_quat"].astype(np.float32)
        g = obs["robot0_gripper_qpos"].astype(np.float32)
        out["gripper"] = np.array([g[0]], dtype=np.float32)
        return out
