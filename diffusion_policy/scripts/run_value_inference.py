if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import copy
import torch
import dill
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np
import h5py
import cv2
import imageio.v2 as imageio

from diffusion_policy.common.pytorch_util import dict_apply


def _load_model_from_checkpoint(ckpt_path, device):
    payload = torch.load(ckpt_path, pickle_module=dill)
    cfg = copy.deepcopy(payload["cfg"])
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(payload["state_dicts"]["model"])
    model.to(device)
    model.eval()
    return cfg, model


def _build_dataset(cfg, dataset_path, use_val):
    if dataset_path is not None:
        cfg.task.dataset_path = dataset_path
        cfg.task.dataset.dataset_path = dataset_path
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if use_val:
        dataset = dataset.get_validation_dataset()
    return dataset


def _get_obs_keys(shape_meta):
    rgb_keys = []
    lowdim_keys = []
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)
        else:
            raise RuntimeError(f"Unsupported obs type: {obs_type}")
    return rgb_keys, lowdim_keys


def _normalize_rgb(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _to_chw_float(img):
    img = _normalize_rgb(img)
    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        img = np.moveaxis(img, -1, 0)
    return img.astype(np.float32) / 255.0


def _load_demo_data(dataset_path, demo, rgb_keys, lowdim_keys, r_key, value_bin_key):
    demo_name = f"demo_{demo}"
    with h5py.File(dataset_path, "r") as f:
        demos = f["data"]
        if demo_name not in demos:
            raise KeyError(f"{demo_name} not found in {dataset_path}")
        demo_group = demos[demo_name]
        if "obs" in demo_group:
            obs_group = demo_group["obs"]
        elif "observations" in demo_group:
            obs_group = demo_group["observations"]
        else:
            raise RuntimeError("No obs group found in demo")

        rgb = {k: obs_group[k][:] for k in rgb_keys}
        lowdim = {k: obs_group[k][:] for k in lowdim_keys}
        r = obs_group[r_key][:]
        value_bin = obs_group[value_bin_key][:] if value_bin_key in obs_group else None

    return rgb, lowdim, r, value_bin


def _run_demo_inference(cfg, model, dataset_path, demo, device, batch_size):
    rgb_keys, lowdim_keys = _get_obs_keys(cfg.shape_meta)
    r_key = cfg.task.dataset.get("r_key", "r")
    value_bin_key = cfg.task.dataset.get("value_bin_key", "value_bin")

    rgb, lowdim, r, value_bin = _load_demo_data(
        dataset_path, demo, rgb_keys, lowdim_keys, r_key, value_bin_key
    )

    length = len(r)
    preds = []
    with torch.no_grad():
        for start in range(0, length, batch_size):
            end = min(start + batch_size, length)
            obs = {}
            for key in rgb_keys:
                imgs = [_to_chw_float(img) for img in rgb[key][start:end]]
                obs[key] = torch.from_numpy(np.stack(imgs, axis=0))
            for key in lowdim_keys:
                obs[key] = torch.from_numpy(
                    lowdim[key][start:end].astype(np.float32)
                )
            obs = dict_apply(obs, lambda x: x.to(device, non_blocking=False))
            logits = model(obs)
            pred_bin = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.append(pred_bin)
    pred_bins = np.concatenate(preds, axis=0)
    return rgb, r, value_bin, pred_bins


def _overlay_text(img_rgb, lines):
    img_bgr = img_rgb[..., ::-1].copy()
    y = 24
    for line in lines:
        cv2.putText(
            img_bgr, line, (10, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        y += 22
    return img_bgr[..., ::-1]

def _draw_line_plot(series, width, height, y_min, y_max, color, title):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if len(series) < 2:
        return canvas

    margin = 24
    x0, y0 = margin, margin
    x1, y1 = width - margin, height - margin
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), 1)

    if y_max == y_min:
        y_max = y_min + 1.0

    xs = np.linspace(x0, x1, num=len(series))
    ys = []
    for v in series:
        t = (v - y_min) / (y_max - y_min)
        t = np.clip(t, 0.0, 1.0)
        y = y1 - t * (y1 - y0)
        ys.append(y)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

    cv2.putText(
        canvas, title, (x0, y0 - 6),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    return canvas


def _make_value_plot_panel(pred_bins, values, width, height):
    if values is None:
        plot = _draw_line_plot(
            pred_bins, width, height,
            y_min=0.0,
            y_max=max(1.0, float(np.max(pred_bins))),
            color=(0, 120, 255),
            title="pred_bin"
        )
        return plot

    half = height // 2
    top = _draw_line_plot(
        pred_bins, width, half,
        y_min=0.0,
        y_max=max(1.0, float(np.max(pred_bins))),
        color=(0, 120, 255),
        title="pred_bin"
    )
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    bottom = _draw_line_plot(
        values, width, height - half,
        y_min=v_min,
        y_max=v_max,
        color=(0, 180, 0),
        title="value"
    )
    return np.vstack([top, bottom])


def _write_video(out_path, frames_rgb, fps):
    out_path = str(out_path)
    try:
        with imageio.get_writer(out_path, fps=fps, format="FFMPEG") as writer:
            for frame in frames_rgb:
                writer.append_data(frame)
        return out_path
    except Exception:
        if not out_path.lower().endswith(".gif"):
            fallback = out_path + ".gif"
        else:
            fallback = out_path
        with imageio.get_writer(fallback, fps=fps) as writer:
            for frame in frames_rgb:
                writer.append_data(frame)
        return fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--dataset-path", default=None, help="Override HDF5 path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Limit number of printed samples (non-demo mode)")
    parser.add_argument("--use-val", action="store_true", help="Use validation split")
    parser.add_argument("--demo", type=int, default=None,
                        help="Run inference on a specific demo index")
    parser.add_argument("--rgb-key", default=None,
                        help="RGB key to visualize (default: first rgb key)")
    parser.add_argument("--output", default="value_inference.mp4",
                        help="Output video path for demo mode")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit frames for demo video")
    parser.add_argument("--plot-values", action="store_true",
                        help="Add value graph panel in demo video")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg, model = _load_model_from_checkpoint(args.ckpt, device)
    dataset_path = args.dataset_path or cfg.task.dataset_path

    if args.demo is not None:
        rgb_keys, _ = _get_obs_keys(cfg.shape_meta)
        if not rgb_keys:
            raise RuntimeError("No RGB keys found for visualization")
        rgb_key = args.rgb_key or rgb_keys[0]

        rgb, r, value_bin, pred_bins = _run_demo_inference(
            cfg, model, dataset_path, args.demo, device, args.batch_size
        )

        if rgb_key not in rgb:
            raise KeyError(f"RGB key {rgb_key} not found in demo data")

        frames = []
        num_frames = len(r)
        limit = num_frames if args.max_frames is None else min(num_frames, args.max_frames)
        for i in range(limit):
            frame = _normalize_rgb(rgb[rgb_key][i])
            lines = [
                f"frame={i}",
                f"pred_bin={int(pred_bins[i])}"
            ]
            if value_bin is not None:
                lines.append(f"target_bin={int(value_bin[i])}")
            if r is not None:
                lines.append(f"value={float(r[i])}")
            frame = _overlay_text(frame, lines)
            if args.plot_values:
                plot_panel = _make_value_plot_panel(
                    pred_bins[: i + 1],
                    r[: i + 1] if r is not None else None,
                    width=max(200, frame.shape[1] // 2),
                    height=frame.shape[0]
                )
                frame = np.concatenate([frame, plot_panel], axis=1)
            frames.append(frame)

        out_path = _write_video(args.output, frames, args.fps)
        print(f"Saved demo video to {out_path}")
        return

    dataset = _build_dataset(cfg, dataset_path, args.use_val)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    num_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=False))
            logits = model(batch["obs"])
            pred_bin = torch.argmax(logits, dim=-1)

            value_bin = batch.get("value_bin", None)
            value = batch.get("value", None)

            batch_size = pred_bin.shape[0]
            for i in range(batch_size):
                if num_seen >= args.num_samples:
                    break
                pred_i = int(pred_bin[i].item())
                if value_bin is not None:
                    target_i = int(value_bin[i].item())
                else:
                    target_i = None
                if value is not None:
                    value_i = float(value[i].item())
                else:
                    value_i = None
                print(f"{num_seen:04d} pred_bin={pred_i} target_bin={target_i} value={value_i}")
                num_seen += 1

            if num_seen >= args.num_samples:
                break


if __name__ == "__main__":
    main()
