"""ItemKNN: косинусная item-item схожесть по взаимодействиям (TF-нормализация, top-K)."""
import numpy as np
from scipy import sparse

from .base import BaseRecommender


class ItemKNNModel(BaseRecommender):
    name = "itemknn"
    display_name = "ItemKNN (cosine)"
    kind = "collaborative"

    def __init__(self, sims: sparse.csr_matrix):
        self.sims = sims.tocsr()

    def scores(self, rated_idx, rvals):
        n = self.sims.shape[0]
        if len(rated_idx) == 0:
            return np.zeros(n, dtype=np.float32)
        w = self.weights(rvals)
        profile = sparse.csr_matrix(
            (w, (np.zeros(len(rated_idx), dtype=np.int32), rated_idx)),
            shape=(1, n),
        )
        out = profile @ self.sims  # (1, n)
        return np.asarray(out.todense()).ravel().astype(np.float32)


def build_sims(item_user: sparse.csr_matrix, top_k: int = 200) -> sparse.csr_matrix:
    """item_user — CSR матрица взаимодействий (items x users) с единичными весами."""
    norms = np.sqrt(item_user.power(2).sum(axis=1)).A.ravel() + 1e-12
    inv = sparse.diags(1.0 / norms)
    normed = inv @ item_user  # косинусная нормировка строк
    sims = normed @ normed.T  # разреженная item x item
    sims = sims.tocsr()
    sims.setdiag(0.0)
    sims.eliminate_zeros()

    # оставляем top-K соседей на строку
    indptr, indices, data = [0], [], []
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz > top_k:
            part = np.argpartition(-row.data, top_k)[:top_k]
            idx = row.indices[part]
            vals = row.data[part]
            order = np.argsort(-vals)
            idx, vals = idx[order], vals[order]
        else:
            idx, vals = row.indices, row.data
        indices.extend(idx.tolist())
        data.extend(vals.tolist())
        indptr.append(len(indices))
    return sparse.csr_matrix((data, indices, indptr), shape=sims.shape, dtype=np.float32)
