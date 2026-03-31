class EvalMetrics:
    def __init__(self):
        self.infer_ms = []
        self.pred_count = 0
        self.kept_count = 0
        self.drop_count = 0
        self.total_reward = 0.0

    def add_infer_ms(self, ms: float):
        self.infer_ms.append(float(ms))

    def add_action_counts(self, pred: int, kept: int, dropped: int):
        self.pred_count += int(pred)
        self.kept_count += int(kept)
        self.drop_count += int(dropped)

    def add_reward(self, r: float):
        self.total_reward += float(r)

    def summary(self):
        avg_infer = sum(self.infer_ms) / len(self.infer_ms) if self.infer_ms else 0.0
        return {
            "avg_infer_ms": avg_infer,
            "pred_count": self.pred_count,
            "kept_count": self.kept_count,
            "drop_count": self.drop_count,
            "return": self.total_reward,
        }
