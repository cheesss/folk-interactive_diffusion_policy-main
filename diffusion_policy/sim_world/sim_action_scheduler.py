from typing import Tuple

import numpy as np


class ActionScheduler:
    """Timestamp-based action filter and cycle slicing."""

    def __init__(self, dt: float, action_offset: int = 0, exec_latency: float = 0.01):
        self.dt = float(dt)
        self.action_offset = int(action_offset)
        self.exec_latency = float(exec_latency)

    def schedule(self, action_seq: np.ndarray, now_time: float) -> Tuple[np.ndarray, int, int]:
        n = len(action_seq)
        ts = (np.arange(n, dtype=np.float64) + self.action_offset) * self.dt + now_time
        keep = ts > (now_time + self.exec_latency)
        kept = action_seq[keep]
        return kept, int(np.sum(~keep)), int(np.sum(keep))

    @staticmethod
    def take_cycle(action_seq: np.ndarray, steps_per_inference: int) -> np.ndarray:
        if len(action_seq) == 0:
            return action_seq
        return action_seq[: max(1, int(steps_per_inference))]
