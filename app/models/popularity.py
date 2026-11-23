"""Baseline: рекомендации по популярности (оценки игнорируются)."""
import numpy as np

from .base import BaseRecommender


class PopularityModel(BaseRecommender):
    name = "popularity"
    display_name = "Popularity"
    kind = "baseline"

    def __init__(self, popularity: np.ndarray):
        self.popularity = popularity.astype(np.float32)

    def scores(self, rated_idx, rvals):
        return self.popularity.copy()
