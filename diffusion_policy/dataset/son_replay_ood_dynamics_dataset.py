"""Expert/rollout HDF5를 OOD dynamics 학습용 샘플로 바꾸는 이미지 데이터셋."""

from typing import Dict, Iterable, List, Sequence
import copy
import hashlib
import json
import os
import shutil
import multiprocessing

import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

register_codecs()


class SonReplayOODDynamicsDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        expert_dataset_path: str = None,
        rollout_dataset_paths: Sequence[str] = tuple(),
        action_horizon: int = 6,
        seed: int = 42,
        val_ratio: float = 0.0,
        abs_action: bool = True,
        rotation_rep: str = "rotation_6d",
        use_cache: bool = False,
    ):
        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1.")

        self.shape_meta = shape_meta
        self.dataset_paths = _resolve_dataset_paths(expert_dataset_path, rollout_dataset_paths)
        self.abs_action = abs_action
        self.seed = seed
        self.val_ratio = val_ratio
        self.rotation_rep = rotation_rep
        self.action_horizon = action_horizon

        rotation_transformer = RotationTransformer(
            from_rep="axis_angle",
            to_rep=rotation_rep,
        )

        if use_cache:
            cache_zarr_path = _get_cache_path(self.dataset_paths)
            cache_lock_path = cache_zarr_path + ".lock"
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    replay_buffer = _convert_hdf5s_to_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_paths=self.dataset_paths,
                        abs_action=abs_action,
                        rotation_transformer=rotation_transformer,
                    )
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                else:
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store,
                            store=zarr.MemoryStore(),
                        )
        else:
            replay_buffer = _convert_hdf5s_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_paths=self.dataset_paths,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )

        rgb_keys = []
        lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=action_horizon + 1,
            pad_before=0,
            pad_after=0,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.action_horizon + 1,
            pad_before=0,
            pad_after=0,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        action_normalizers = [
            get_range_normalizer_from_stat(
                array_to_stats(self.replay_buffer["action"][..., :3])
            ),
            get_identity_normalizer_from_stat(
                array_to_stats(self.replay_buffer["action"][..., 3:9])
            ),
            get_range_normalizer_from_stat(
                array_to_stats(self.replay_buffer["action"][..., 9:10])
            ),
        ]
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith("position"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("gripper"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported lowdim key {key}")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        obs_dict = {}
        next_obs_dict = {}
        for key in self.rgb_keys:
            current = np.moveaxis(data[key][0], -1, 0).astype(np.float32) / 255.0
            nxt = np.moveaxis(data[key][-1], -1, 0).astype(np.float32) / 255.0
            obs_dict[key] = current
            next_obs_dict[key] = nxt
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][0].astype(np.float32)
            next_obs_dict[key] = data[key][-1].astype(np.float32)
            del data[key]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "next_obs": dict_apply(next_obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"][: self.action_horizon].astype(np.float32)),
        }


def _resolve_dataset_paths(
    expert_dataset_path: str,
    rollout_dataset_paths: Iterable[str],
) -> List[str]:
    dataset_paths = []
    if expert_dataset_path:
        dataset_paths.append(expert_dataset_path)
    dataset_paths.extend(list(rollout_dataset_paths or []))
    if len(dataset_paths) == 0:
        raise ValueError("At least one dataset path is required.")
    return dataset_paths


def _get_cache_path(dataset_paths: Sequence[str]) -> str:
    payload = json.dumps(sorted(dataset_paths)).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    base_dir = os.path.dirname(dataset_paths[0]) or "."
    return os.path.join(base_dir, f".ood_dynamics_cache_{digest}.zarr.zip")


def _get_obs_group(demo):
    if "obs" in demo:
        return demo["obs"]
    if "observations" in demo:
        return demo["observations"]
    raise RuntimeError("No obs or observations group found in demo.")


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        if raw_actions.shape[-1] == 10:
            return raw_actions.astype(np.float32)
        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    return actions


def _convert_hdf5s_to_replay(
    store,
    shape_meta,
    dataset_paths,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys = []
    lowdim_keys = []
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episode_ends = []
    episode_starts = []
    prev_end = 0
    total_steps = 0
    episode_specs = []
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as file:
            demos = file["data"]
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]
                episode_length = demo["actions"].shape[0]
                episode_starts.append(prev_end)
                prev_end += episode_length
                episode_ends.append(prev_end)
                total_steps += episode_length
                episode_specs.append((dataset_path, i))

    _ = meta_group.array(
        "episode_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        overwrite=True,
    )

    for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
        this_data = []
        for dataset_path in dataset_paths:
            with h5py.File(dataset_path, "r") as file:
                demos = file["data"]
                for i in range(len(demos)):
                    demo = demos[f"demo_{i}"]
                    obs_group = _get_obs_group(demo)
                    if key == "action":
                        arr = demo["actions"][:].astype(np.float32)
                    else:
                        arr = obs_group[key][:].astype(np.float32)
                    this_data.append(arr)
        this_data = np.concatenate(this_data, axis=0)
        if key == "action":
            this_data = _convert_actions(
                raw_actions=this_data,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

    with tqdm(total=total_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
        for key in rgb_keys:
            c, h, w = tuple(shape_meta["obs"][key]["shape"])
            img_arr = data_group.require_dataset(
                name=key,
                shape=(total_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=Jpeg2k(level=50),
                dtype=np.uint8,
            )
            for episode_idx, (dataset_path, demo_idx) in enumerate(episode_specs):
                with h5py.File(dataset_path, "r") as file:
                    demo = file["data"][f"demo_{demo_idx}"]
                    obs_group = _get_obs_group(demo)
                    hdf5_arr = obs_group[key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        img_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                        _ = img_arr[zarr_idx]
                        pbar.update(1)

    return ReplayBuffer(root)
