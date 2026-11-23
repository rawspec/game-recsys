"""Гибрид: ансамбль контентной модели и ALS (вес ALS растёт с числом оценок)."""
import numpy as np

from .als import ALSModel
from .base import BaseRecommender
from .content import ContentModel


class HybridModel(BaseRecommender):
    name = "hybrid"
    display_name = "Hybrid (content + ALS)"
    kind = "hybrid"

    def __init__(self, content: ContentModel, als: ALSModel,
                 popularity: np.ndarray, w_als_max: float = 0.5,
                 n_warmup: int = 15, pop_prior: float = 0.08):
        self.content = content
        self.als = als
        self.popularity = popularity.astype(np.float32)
        self.w_als_max = w_als_max
        self.n_warmup = n_warmup
        self.pop_prior = pop_prior

    def scores(self, rated_idx, rvals):
        s_content = self.content.normalized(rated_idx, rvals)
        if len(rated_idx) == 0:
            return self.popularity.copy()
        s_als = self.als.normalized(rated_idx, rvals)
        w_als = self.w_als_max * min(1.0, len(rated_idx) / self.n_warmup)
        score = (1.0 - w_als) * s_content + w_als * s_als
        return (1.0 - self.pop_prior) * score + self.pop_prior * self.popularity
