"""Единый интерфейс рекомендательных моделей.

Каждая модель умеет:
  scores(rated_idx, rvals) -> np.ndarray[n_games]  — сырой скор для всех игр
где rated_idx — индексы оценённых игр, rvals — оценки 1..10 (могут быть пустыми).
"""
from __future__ import annotations

import numpy as np


class BaseRecommender:
    name: str = "base"
    display_name: str = "Base"
    kind: str = "collaborative"  # collaborative | content | hybrid | baseline

    def scores(self, rated_idx: np.ndarray, rvals: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def weights(self, rvals: np.ndarray) -> np.ndarray:
        """Вес оценки в профиле пользователя: 1..10 -> [-1, 1]."""
        w = (np.asarray(rvals, dtype=np.float32) - 5.5) / 4.5
        if w.size and np.all(np.abs(w) < 1e-6):
            w = np.asarray(rvals, dtype=np.float32) / 10.0
        return w

    def normalized(self, rated_idx: np.ndarray, rvals: np.ndarray) -> np.ndarray:
        """Сырой скор, отнормированный в [0, 1] робастно (5–95 перцентили)."""
        raw = self.scores(rated_idx, rvals).astype(np.float64)
        if raw.size == 0:
            return raw.astype(np.float32)
        lo, hi = np.percentile(raw, [5.0, 95.0])
        span = float(max(hi - lo, 1e-9))
        return np.clip((raw - lo) / span, 0.0, 1.0).astype(np.float32)
