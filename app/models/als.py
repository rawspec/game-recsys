"""ALS (implicit feedback) с fold-in нового пользователя."""
import numpy as np

from .base import BaseRecommender


class ALSModel(BaseRecommender):
    name = "als"
    display_name = "ALS (implicit)"
    kind = "collaborative"

    def __init__(self, item_factors: np.ndarray, reg: float, fold_in_reg: float | None = None):
        self.Y = item_factors.astype(np.float32)
        self.reg = float(reg)
        # fold-in сильно недорегуляризован при малом λ — по умолчанию усиливаем
        self.fold_in_reg = float(fold_in_reg if fold_in_reg is not None else max(reg, 100.0))
        self.f = self.Y.shape[1]

    def scores(self, rated_idx, rvals):
        if len(rated_idx) == 0:
            return np.zeros(self.Y.shape[0], dtype=np.float32)
        c = np.asarray(rvals, dtype=np.float32)
        Yr = self.Y[rated_idx]
        A = (Yr.T * c) @ Yr + self.fold_in_reg * np.eye(self.f, dtype=np.float32)
        b = Yr.T @ c
        x = np.linalg.solve(A, b)
        return self.Y @ x
