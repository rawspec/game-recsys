"""LightFM: гибридная матричная факторизация (обёртка над опциональной зависимостью)."""
import numpy as np

from .base import BaseRecommender


class LightFMModel(BaseRecommender):
    name = "lightfm"
    display_name = "LightFM (WARP)"
    kind = "hybrid"

    def __init__(self, item_embeddings: np.ndarray, item_biases: np.ndarray):
        self.V = item_embeddings.astype(np.float32)
        self.b = item_biases.astype(np.float32)

    def scores(self, rated_idx, rvals):
        if len(rated_idx) == 0:
            return np.zeros(self.V.shape[0], dtype=np.float32)
        w = np.clip(self.weights(rvals), 0.0, 1.0)
        if w.sum() < 1e-9:
            w = np.ones(len(rated_idx), dtype=np.float32)
        u = (self.V[rated_idx] * w[:, None]).sum(axis=0) / w.sum()
        return self.V @ u + self.b
