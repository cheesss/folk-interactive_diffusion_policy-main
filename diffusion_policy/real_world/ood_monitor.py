"""실시간 inference/teleop 중 current/predicted OOD 점수를 계산하는 런타임 모듈."""

from typing import Dict, Optional

import dill
import hydra
import torch

from diffusion_policy.common.ood_utils import (
    chunked_min_l2,
    encode_policy_obs,
    normalize_action,
    normalize_ood_score,
)


class OODMonitor:
    """Expert latent bank와 optional dynamics model을 묶어 OOD 점수를 계산한다."""
    def __init__(
        self,
        policy,
        bank_path: str,
        expected_shape_meta: Optional[dict] = None,
        dynamics_ckpt: Optional[str] = None,
        threshold: Optional[float] = None,
        device: str = "cuda",
        chunk_size: int = 4096,
    ):
        self.policy = policy
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        bank_payload = torch.load(bank_path, map_location=self.device)
        self.reference_bank = bank_payload["latents"].to(self.device)
        self.reference_stats = bank_payload["stats"]
        self.lowdim_keys = bank_payload["lowdim_keys"]
        self.bank_shape_meta = bank_payload.get("shape_meta")
        if expected_shape_meta is not None and self.bank_shape_meta is not None:
            _validate_shape_meta_compat(self.bank_shape_meta, expected_shape_meta)
        self.threshold_raw = threshold if threshold is not None else self.reference_stats.get("p95", 1.0)
        self.threshold_normalized = float(
            normalize_ood_score(torch.tensor(self.threshold_raw), self.reference_stats).item()
        )

        self.dynamics_model = None
        if dynamics_ckpt is not None:
            payload = torch.load(open(dynamics_ckpt, "rb"), pickle_module=dill, map_location=self.device)
            cfg = payload["cfg"]
            self.dynamics_model = hydra.utils.instantiate(cfg.model)
            self.dynamics_model.load_state_dict(payload["state_dicts"]["model"])
            self.dynamics_model.to(self.device).eval()

    def score_current(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            latent, proprio = encode_policy_obs(self.policy, obs_dict, self.lowdim_keys)
            raw = chunked_min_l2(latent, self.reference_bank, self.chunk_size)
            normalized = normalize_ood_score(raw, self.reference_stats)
        return {
            "raw": float(raw[0].item()),
            "normalized": float(normalized[0].item()),
            "latent": latent,
            "proprio": proprio,
        }

    def score_predicted(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_seq: torch.Tensor,
    ) -> Optional[Dict[str, float]]:
        if self.dynamics_model is None:
            return None

        with torch.no_grad():
            latent, proprio = encode_policy_obs(self.policy, obs_dict, self.lowdim_keys)
            if action_seq.ndim == 2:
                action_seq = action_seq.unsqueeze(0)
            norm_action_seq = normalize_action(self.policy, action_seq)
            pred_latent, pred_proprio = self.dynamics_model(latent, proprio, norm_action_seq)
            raw = chunked_min_l2(pred_latent, self.reference_bank, self.chunk_size)
            normalized = normalize_ood_score(raw, self.reference_stats)

        return {
            "raw": float(raw[0].item()),
            "normalized": float(normalized[0].item()),
            "future_proprio": pred_proprio[0].detach().cpu().tolist(),
        }


def _validate_shape_meta_compat(reference_shape_meta: dict, expected_shape_meta: dict):
    ref_obs = reference_shape_meta["obs"]
    exp_obs = expected_shape_meta["obs"]

    ref_keys = set(ref_obs.keys())
    exp_keys = set(exp_obs.keys())
    missing = sorted(exp_keys - ref_keys)
    extra = sorted(ref_keys - exp_keys)
    if missing or extra:
        raise ValueError(
            "OOD bank shape_meta mismatch. "
            f"missing_in_bank={missing}, extra_in_bank={extra}"
        )

    for key in sorted(exp_keys):
        ref_attr = ref_obs[key]
        exp_attr = exp_obs[key]
        ref_type = ref_attr.get("type", "low_dim")
        exp_type = exp_attr.get("type", "low_dim")
        ref_shape = tuple(ref_attr["shape"])
        exp_shape = tuple(exp_attr["shape"])
        if ref_type != exp_type or ref_shape != exp_shape:
            raise ValueError(
                f"OOD bank shape_meta mismatch for key '{key}': "
                f"bank(type={ref_type}, shape={ref_shape}) vs "
                f"runtime(type={exp_type}, shape={exp_shape})"
            )
