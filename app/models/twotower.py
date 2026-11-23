"""Two-Tower (PyTorch): user-tower = MLP от взвешенного среднего item-эмбеддингов.

На инференсе сохранённые numpy-веса MLP повторяются без torch.
"""
import numpy as np

from .base import BaseRecommender


class TwoTowerModel(BaseRecommender):
    name = "twotower"
    display_name = "Two-Tower (DL)"
    kind = "collaborative"

    def __init__(self, item_emb: np.ndarray, w1: np.ndarray, b1: np.ndarray,
                 w2: np.ndarray, b2: np.ndarray):
        self.E = item_emb.astype(np.float32)
        self.w1, self.b1, self.w2, self.b2 = (
            w1.astype(np.float32), b1.astype(np.float32),
            w2.astype(np.float32), b2.astype(np.float32),
        )

    def _user_vec(self, rated_idx, rvals):
        w = np.clip(self.weights(rvals), 0.0, 1.0)  # только положительные сигналы
        if w.sum() < 1e-9:
            w = np.ones(len(rated_idx), dtype=np.float32)
        z = (self.E[rated_idx] * w[:, None]).sum(axis=0) / w.sum()
        h = np.maximum(z @ self.w1 + self.b1, 0.0)
        u = h @ self.w2 + self.b2
        return u / max(np.linalg.norm(u), 1e-9)

    def scores(self, rated_idx, rvals):
        if len(rated_idx) == 0:
            return np.zeros(self.E.shape[0], dtype=np.float32)
        u = self._user_vec(rated_idx, rvals)
        return self.E @ u
