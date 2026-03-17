"""Expert demo를 latent bank(.pt)로 내보내는 오프라인 export 스크립트."""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib

import dill
import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.ood_utils import (
    compute_reference_stats,
    encode_policy_obs,
    get_lowdim_keys,
)


def _get_obs_group(demo):
    if "obs" in demo:
        return demo["obs"]
    if "observations" in demo:
        return demo["observations"]
    raise RuntimeError("No obs or observations group found in demo.")


def _load_policy_from_checkpoint(ckpt_path: str, use_ema: bool):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    if use_ema and getattr(workspace, "ema_model", None) is not None:
        return workspace.ema_model
    return workspace.model


def _iter_obs_batches(dataset_path, shape_meta, batch_size, device):
    lowdim_keys = get_lowdim_keys(shape_meta)
    rgb_keys = [key for key, attr in shape_meta["obs"].items() if attr.get("type", "low_dim") == "rgb"]

    with h5py.File(dataset_path, "r") as file:
        demos = file["data"]
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            obs_group = _get_obs_group(demo)
            length = obs_group[rgb_keys[0]].shape[0] if rgb_keys else obs_group[lowdim_keys[0]].shape[0]
            for start in range(0, length, batch_size):
                end = min(start + batch_size, length)
                obs = {}
                for key in rgb_keys:
                    arr = obs_group[key][start:end]
                    if arr.dtype != np.uint8:
                        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    arr = torch.from_numpy(arr).to(device)
                    arr = arr.permute(0, 3, 1, 2).float() / 255.0
                    obs[key] = arr
                for key in lowdim_keys:
                    obs[key] = torch.from_numpy(obs_group[key][start:end].astype("float32")).to(device)
                yield obs


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="son_export_ood_assets",
)
def main(cfg: OmegaConf):
    device = torch.device(cfg.export.device)
    policy = _load_policy_from_checkpoint(
        cfg.reference_policy.checkpoint,
        cfg.reference_policy.use_ema,
    )
    policy.to(device).eval()
    for param in policy.parameters():
        param.requires_grad = False

    lowdim_keys = get_lowdim_keys(cfg.task.shape_meta)
    latents = []
    with torch.no_grad():
        for obs in _iter_obs_batches(
            cfg.task.expert_dataset_path,
            cfg.task.shape_meta,
            cfg.export.batch_size,
            device,
        ):
            latent, _ = encode_policy_obs(policy, obs, lowdim_keys)
            latents.append(latent.detach().cpu())

    latents = torch.cat(latents, dim=0)
    stats = compute_reference_stats(latents.to(device), chunk_size=cfg.export.chunk_size)

    output_path = pathlib.Path(cfg.export.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": latents,
            "stats": stats,
            "lowdim_keys": lowdim_keys,
            "shape_meta": OmegaConf.to_container(cfg.task.shape_meta, resolve=True),
            "reference_policy_checkpoint": cfg.reference_policy.checkpoint,
            "expert_dataset_path": cfg.task.expert_dataset_path,
        },
        output_path,
    )
    print(f"Saved OOD reference bank to {output_path}")


if __name__ == "__main__":
    main()
