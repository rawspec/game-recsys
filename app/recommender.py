"""Оркестратор рекомендаций: кэш ранжирования на каждый алгоритм."""
import numpy as np

from .models.registry import ModelRegistry


class Recommender:
    def __init__(self, store) -> None:
        self.store = store
        self.registry = ModelRegistry()
        self._caches: dict[str, dict] = {}

    def _recompute(self, algo: str, ratings: dict, hidden: set) -> dict:
        st = self.store
        model = self.registry.get(algo)
        n = len(st.appids)
        rated_idx = np.array(
            [st.appid_to_idx[a] for a in ratings if a in st.appid_to_idx],
            dtype=np.int64,
        )
        rvals = np.array([ratings[int(st.appids[i])] for i in rated_idx],
                         dtype=np.float32)
        score = model.normalized(rated_idx, rvals)

        excluded = np.zeros(n, dtype=bool)
        if len(rated_idx):
            excluded[rated_idx] = True
        for appid in hidden:
            idx = st.appid_to_idx.get(int(appid))
            if idx is not None:
                excluded[idx] = True
        masked = np.where(excluded, -1.0, score)
        ranked = np.argsort(-masked, kind="stable")
        return {"ranked": ranked[: int((~excluded).sum())], "scores": score}

    def page(self, offset: int, limit: int, algo: str = "hybrid") -> dict:
        st = self.store
        if self.registry.get(algo) is None:
            algo = "hybrid"
        version = st.version()
        cache = self._caches.get(algo)
        if not cache or cache["version"] != version:
            ratings = st.get_ratings()
            hidden = st.get_hidden()
            cache = self._recompute(algo, ratings, hidden)
            cache["version"] = version
            cache["ratings"] = ratings
            self._caches[algo] = cache

        ranked = cache["ranked"]
        slice_idx = ranked[offset:offset + limit]
        ratings = cache["ratings"]
        reasons = self._reasons(slice_idx, ratings) if ratings else [None] * len(slice_idx)
        items = [
            st.game_json(
                int(idx),
                rating=ratings.get(int(st.appids[idx])),
                score=cache["scores"][idx],
                reason=reason,
            )
            for idx, reason in zip(slice_idx, reasons)
        ]
        return {"total": len(ranked), "offset": offset, "algo": algo, "items": items}

    def _reasons(self, page_idx, ratings) -> list:
        st = self.store
        rated_idx = np.array(
            [st.appid_to_idx[a] for a in ratings if a in st.appid_to_idx],
            dtype=np.int64,
        )
        if len(rated_idx) == 0 or len(page_idx) == 0:
            return [None] * len(page_idx)
        rvals = np.array([ratings[int(st.appids[i])] for i in rated_idx],
                         dtype=np.float32)
        sims = st.embeddings[page_idx] @ st.embeddings[rated_idx].T
        contrib = sims * (rvals / 10.0)[None, :]
        best = contrib.argmax(axis=1)
        return [f"Похоже на «{st.names[rated_idx[b]]}»" for b in best]
