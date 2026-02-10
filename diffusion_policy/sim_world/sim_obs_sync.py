from collections import deque
from typing import Dict

import numpy as np


class ObsWindowSync:
    """Build fixed-length observation windows for policy input."""

    def __init__(self, n_obs_steps: int):
        self.n_obs_steps = int(n_obs_steps)
        self.buffer = deque(maxlen=max(1, self.n_obs_steps))

    def reset(self, first_obs: Dict[str, np.ndarray]):
        self.buffer.clear()
        for _ in range(self.n_obs_steps):
            self.buffer.append({k: np.array(v, copy=True) for k, v in first_obs.items()})

    def push(self, obs: Dict[str, np.ndarray]):
        self.buffer.append({k: np.array(v, copy=True) for k, v in obs.items()})

    def get_window(self) -> Dict[str, np.ndarray]:
        assert len(self.buffer) > 0
        keys = self.buffer[0].keys()
        out = {}
        for k in keys:
            out[k] = np.stack([x[k] for x in self.buffer], axis=0)
        return out
