if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import copy
import h5py
import numpy as np
import torch
import dill
import hydra

from diffusion_policy.common.pytorch_util import dict_apply


def _load_value_model(ckpt_path, device):
    payload = torch.load(ckpt_path, pickle_module=dill)
    cfg = copy.deepcopy(payload["cfg"])
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(payload["state_dicts"]["model"])
    model.to(device).eval()
    return cfg, model


def _get_shape_meta(cfg):
    if hasattr(cfg, "task") and hasattr(cfg.task, "shape_meta"):
        return cfg.task.shape_meta
    if hasattr(cfg, "shape_meta"):
        return cfg.shape_meta
    if hasattr(cfg, "model") and hasattr(cfg.model, "shape_meta"):
        return cfg.model.shape_meta
    raise RuntimeError("shape_meta not found in checkpoint config")


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


def _compute_value_norm(model, device, rgb, lowdim, rgb_keys, lowdim_keys, bin_centers, batch_size):
    preds = []
    with torch.no_grad():
        for start in range(0, rgb[rgb_keys[0]].shape[0], batch_size):
            end = start + batch_size
            obs = {}
            for key in rgb_keys:
                imgs = [_to_chw_float(img) for img in rgb[key][start:end]]
                obs[key] = torch.from_numpy(np.stack(imgs, axis=0))
            for key in lowdim_keys:
                obs[key] = torch.from_numpy(lowdim[key][start:end].astype(np.float32))
            obs = dict_apply(obs, lambda x: x.to(device, non_blocking=False))
            logits = model(obs)
            probs = torch.softmax(logits, dim=-1)
            v = (probs * bin_centers).sum(dim=-1)
            preds.append(v.cpu().numpy())
    return np.concatenate(preds, axis=0)


def _get_obs_group(demo_group):
    if "obs" in demo_group:
        return demo_group["obs"]
    if "observations" in demo_group:
        return demo_group["observations"]
    raise RuntimeError("No obs group found in demo")


def _delete_if_exists(group, name):
    if name in group:
        del group[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--ckpt", required=True, help="Value model checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--r-key", default="r")
    parser.add_argument("--bin-min", type=float, default=-1.0)
    parser.add_argument("--bin-max", type=float, default=0.0)
    parser.add_argument("--adv-key", default="advantage")
    parser.add_argument("--adv-ind-key", default="advantage_indicator")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg, model = _load_value_model(args.ckpt, device)
    shape_meta = _get_shape_meta(cfg)
    rgb_keys, lowdim_keys = _get_obs_keys(shape_meta)
    if len(rgb_keys) == 0:
        raise RuntimeError("No RGB keys found in shape_meta")

    num_bins = int(getattr(model, "num_bins", 201))
    bin_centers = torch.linspace(args.bin_min, args.bin_max, steps=num_bins, device=device)

    with h5py.File(args.hdf5, "r+") as f:
        demos = f["data"]
        demo_names = sorted(list(demos.keys()))
        lengths = []
        for name in demo_names:
            obs_group = _get_obs_group(demos[name])
            lengths.append(len(obs_group[args.r_key]))
        t_max = max(lengths)
        if t_max <= 0:
            raise RuntimeError("Invalid max episode length")

        all_adv = []
        for name in demo_names:
            demo = demos[name]
            obs_group = _get_obs_group(demo)
            r = obs_group[args.r_key][:].astype(np.float32)
            rgb = {k: obs_group[k][:] for k in rgb_keys}
            lowdim = {k: obs_group[k][:] for k in lowdim_keys}

            v_norm = _compute_value_norm(
                model, device, rgb, lowdim, rgb_keys, lowdim_keys,
                bin_centers, args.batch_size
            )

            ret = np.cumsum(r[::-1], axis=0)[::-1]
            remaining_steps = np.arange(len(r), 0, -1, dtype=np.float32)
            ret_norm = ret / np.maximum(remaining_steps, 1.0)
            adv = ret_norm - v_norm

            _delete_if_exists(obs_group, args.adv_key)
            obs_group.create_dataset(args.adv_key, data=adv.astype(np.float32), compression="gzip")
            all_adv.append(adv.astype(np.float32))

        all_adv = np.concatenate(all_adv, axis=0)
        thresh = np.quantile(all_adv, 0.7)

        for name in demo_names:
            demo = demos[name]
            obs_group = _get_obs_group(demo)
            adv = obs_group[args.adv_key][:]
            ind = (adv >= thresh).astype(np.uint8)
            _delete_if_exists(obs_group, args.adv_ind_key)
            obs_group.create_dataset(args.adv_ind_key, data=ind, compression="gzip")

    print(f"Saved {args.adv_key} and {args.adv_ind_key} to {args.hdf5}")
    print(f"Threshold (70th percentile): {thresh:.6f}")


if __name__ == "__main__":
    main()
