from typing import Dict, List, Optional
import numpy as np
import h5py
import torch

from diffusion_policy.dataset.base_dataset import BaseImageDataset


class SonReplayValueImageDataset(BaseImageDataset):
    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            r_key: str = 'r',
            value_bin_key: str = 'value_bin',
            val_ratio: float = 0.0,
            seed: int = 42,
            _cache: Optional[dict] = None,
            indices: Optional[np.ndarray] = None):
        self.shape_meta = shape_meta
        self.dataset_path = dataset_path
        self.r_key = r_key
        self.value_bin_key = value_bin_key

        obs_shape_meta = shape_meta['obs']
        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                self.rgb_keys.append(key)
            elif obs_type == 'low_dim':
                self.lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        if _cache is None:
            self._cache = self._load_hdf5()
        else:
            self._cache = _cache

        n_samples = len(self._cache['r'])
        if indices is None:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_samples)
            n_val = int(n_samples * val_ratio)
            self.indices = perm[n_val:]
            self._val_indices = perm[:n_val]
        else:
            self.indices = indices
            self._val_indices = np.array([], dtype=np.int64)

    def _load_hdf5(self) -> dict:
        rgb_data = {k: [] for k in self.rgb_keys}
        lowdim_data = {k: [] for k in self.lowdim_keys}
        r_list: List[np.ndarray] = []
        value_bin_list: List[np.ndarray] = []
        has_value_bin = True

        with h5py.File(self.dataset_path, 'r') as f:
            demos = f['data']
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                if 'obs' in demo:
                    obs_group = demo['obs']
                elif 'observations' in demo:
                    obs_group = demo['observations']
                else:
                    raise RuntimeError("No obs group found in demo")

                for key in self.rgb_keys:
                    rgb_data[key].append(obs_group[key][:])
                for key in self.lowdim_keys:
                    lowdim_data[key].append(obs_group[key][:])
                if self.r_key not in obs_group:
                    raise RuntimeError(f"Missing r key {self.r_key} in obs")
                r_list.append(obs_group[self.r_key][:])
                if self.value_bin_key in obs_group:
                    value_bin_list.append(obs_group[self.value_bin_key][:])
                else:
                    has_value_bin = False

        rgb_data = {k: np.concatenate(v, axis=0) for k, v in rgb_data.items()}
        lowdim_data = {k: np.concatenate(v, axis=0) for k, v in lowdim_data.items()}
        r_data = np.concatenate(r_list, axis=0).astype(np.float32)
        value_bin_data = None
        if has_value_bin and value_bin_list:
            value_bin_data = np.concatenate(value_bin_list, axis=0).astype(np.int64)

        cache = {
            'rgb': rgb_data,
            'lowdim': lowdim_data,
            'r': r_data
        }
        if value_bin_data is not None:
            cache['value_bin'] = value_bin_data
        return cache

    def get_validation_dataset(self) -> 'SonReplayValueImageDataset':
        return SonReplayValueImageDataset(
            shape_meta=self.shape_meta,
            dataset_path=self.dataset_path,
            r_key=self.r_key,
            value_bin_key=self.value_bin_key,
            _cache=self._cache,
            indices=self._val_indices
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        obs_dict = dict()

        for key in self.rgb_keys:
            img = self._cache['rgb'][key][i]
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img = np.moveaxis(img, -1, 0).astype(np.float32) / 255.0
            obs_dict[key] = img

        for key in self.lowdim_keys:
            obs_dict[key] = self._cache['lowdim'][key][i].astype(np.float32)

        value = self._cache['r'][i].astype(np.float32)
        value_bin = None
        if 'value_bin' in self._cache:
            value_bin = self._cache['value_bin'][i].astype(np.int64)

        batch = {
            'obs': {k: torch.from_numpy(v) for k, v in obs_dict.items()},
            'value': torch.tensor(value, dtype=torch.float32)
        }
        if value_bin is not None:
            batch['value_bin'] = torch.tensor(value_bin, dtype=torch.long)
        return batch
