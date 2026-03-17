"""OOD 계산에서 공통으로 쓰는 관측 인코딩, 거리 계산, 점수 정규화 유틸리티."""

from typing import Dict, List, Tuple

import torch

from diffusion_policy.common.pytorch_util import dict_apply


def get_lowdim_keys(shape_meta: dict) -> List[str]:
    keys = []
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") == "low_dim":
            keys.append(key)
    return keys


def add_time_dim(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return dict_apply(obs_dict, lambda x: x.unsqueeze(1))


def stack_lowdim(obs_dict: Dict[str, torch.Tensor], lowdim_keys: List[str]) -> torch.Tensor:
    return torch.cat([obs_dict[key] for key in lowdim_keys], dim=-1)


def encode_policy_obs(
    policy,
    obs_dict: Dict[str, torch.Tensor],
    lowdim_keys: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Policy normalizer/encoder expect a time dimension, even for single-step inputs.
    obs_seq = add_time_dim(obs_dict)
    nobs = policy.normalizer.normalize(obs_seq)
    encoder_in = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
    latent = policy.obs_encoder(encoder_in)
    proprio = stack_lowdim({k: nobs[k][:, 0] for k in lowdim_keys}, lowdim_keys)
    return latent, proprio


def normalize_action(policy, action: torch.Tensor) -> torch.Tensor:
    return policy.normalizer["action"].normalize(action)


def chunked_min_l2(
    query: torch.Tensor,
    bank: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    mins = []
    for start in range(0, query.shape[0], chunk_size):
        q = query[start:start + chunk_size]
        best = None
        for bank_start in range(0, bank.shape[0], chunk_size):
            b = bank[bank_start:bank_start + chunk_size]
            dist = torch.cdist(q, b, p=2)
            min_dist = dist.min(dim=-1).values
            if best is None:
                best = min_dist
            else:
                best = torch.minimum(best, min_dist)
        mins.append(best)
    return torch.cat(mins, dim=0)


def compute_reference_stats(
    bank: torch.Tensor,
    chunk_size: int = 2048,
) -> Dict[str, float]:
    if bank.shape[0] < 2:
        return {
            "median": 0.0,
            "mean": 0.0,
            "std": 1.0,
            "p95": 1.0,
            "p99": 1.0,
        }

    distances = []
    for start in range(0, bank.shape[0], chunk_size):
        q = bank[start:start + chunk_size]
        best = None
        for bank_start in range(0, bank.shape[0], chunk_size):
            b = bank[bank_start:bank_start + chunk_size]
            dist = torch.cdist(q, b, p=2)
            if start == bank_start:
                diag_len = min(dist.shape[0], dist.shape[1])
                idx = torch.arange(diag_len, device=dist.device)
                dist[idx, idx] = float("inf")
            min_dist = dist.min(dim=-1).values
            if best is None:
                best = min_dist
            else:
                best = torch.minimum(best, min_dist)
        distances.append(best)

    distances = torch.cat(distances, dim=0)
    return {
        "median": float(torch.quantile(distances, 0.5).item()),
        "mean": float(distances.mean().item()),
        "std": float(distances.std(unbiased=False).item() + 1e-6),
        "p95": float(torch.quantile(distances, 0.95).item()),
        "p99": float(torch.quantile(distances, 0.99).item()),
    }


def normalize_ood_score(score: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    median = stats.get("median", 0.0)
    p95 = stats.get("p95", 1.0)
    denom = max(p95 - median, 1e-6)
    return torch.clamp((score - median) / denom, min=0.0)
