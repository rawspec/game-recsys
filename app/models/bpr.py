"""BPR-MF: байесовская персонализированная ранжирующая матричная факторизация."""
import numpy as np

from .base import BaseRecommender


class BPRModel(BaseRecommender):
    name = "bpr"
    display_name = "BPR-MF"
    kind = "collaborative"

    def __init__(self, item_factors: np.ndarray):
        self.V = item_factors.astype(np.float32)

    def scores(self, rated_idx, rvals):
        if len(rated_idx) == 0:
            return np.zeros(self.V.shape[0], dtype=np.float32)
        w = self.weights(rvals)
        u = (self.V[rated_idx] * w[:, None]).sum(axis=0)
        u /= max(np.linalg.norm(u), 1e-9)
        return self.V @ u
