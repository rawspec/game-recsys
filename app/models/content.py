"""Контентная модель: косинусная близость к профилю пользователя из эмбеддингов игр."""
import numpy as np

from .base import BaseRecommender


class ContentModel(BaseRecommender):
    name = "content"
    display_name = "Content (MiniLM)"
    kind = "content"

    def __init__(self, embeddings: np.ndarray):
        self.E = embeddings.astype(np.float32)

    def scores(self, rated_idx, rvals):
        if len(rated_idx) == 0:
            return np.zeros(self.E.shape[0], dtype=np.float32)
        w = self.weights(rvals)
        u = (self.E[rated_idx] * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(u)
        if norm < 1e-9:
            u = self.E[rated_idx].mean(axis=0)
            norm = np.linalg.norm(u)
        u /= max(norm, 1e-9)
        return self.E @ u
