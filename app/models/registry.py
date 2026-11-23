"""Реестр моделей: загрузка артефактов и выбор алгоритма."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .als import ALSModel
from .bpr import BPRModel
from .content import ContentModel
from .hybrid import HybridModel
from .itemknn import ItemKNNModel
from .lightfm import LightFMModel
from .popularity import PopularityModel
from .twotower import TwoTowerModel

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class ModelRegistry:
    def __init__(self) -> None:
        self.popularity = np.load(PROC_DIR / "popularity.npy")
        self._models = {"popularity": PopularityModel(self.popularity)}

        content = ContentModel(np.load(PROC_DIR / "embeddings.npy"))
        self._models["content"] = content

        meta = json.loads((PROC_DIR / "als_meta.json").read_text(encoding="utf-8"))
        als = ALSModel(np.load(PROC_DIR / "als_item_factors.npy"), meta["reg"],
                       meta.get("fold_in_reg"))
        self._models["als"] = als
        self._models["hybrid"] = HybridModel(content, als, self.popularity)

        optional = {
            "itemknn": (MODELS_DIR / "itemknn_sims.npz",
                        lambda: ItemKNNModel(sparse.load_npz(MODELS_DIR / "itemknn_sims.npz"))),
            "bpr": (MODELS_DIR / "bpr_item_factors.npy",
                    lambda: BPRModel(np.load(MODELS_DIR / "bpr_item_factors.npy"))),
            "twotower": (MODELS_DIR / "twotower_item.npy",
                         lambda: TwoTowerModel(
                             np.load(MODELS_DIR / "twotower_item.npy"),
                             np.load(MODELS_DIR / "twotower_w1.npy"),
                             np.load(MODELS_DIR / "twotower_b1.npy"),
                             np.load(MODELS_DIR / "twotower_w2.npy"),
                             np.load(MODELS_DIR / "twotower_b2.npy"),
                         )),
            "lightfm": (MODELS_DIR / "lightfm_item.npy",
                        lambda: LightFMModel(
                            np.load(MODELS_DIR / "lightfm_item.npy"),
                            np.load(MODELS_DIR / "lightfm_bias.npy"),
                        )),
        }
        for name, (path, loader) in optional.items():
            if Path(path).exists():
                try:
                    self._models[name] = loader()
                except Exception:
                    pass  # битый артефакт — пропускаем, можно переобучить

    def get(self, name: str):
        return self._models.get(name)

    def names(self) -> list[str]:
        return list(self._models.keys())

    def infos(self) -> list[dict]:
        return [
            {"name": m.name, "display_name": m.display_name, "kind": m.kind}
            for m in self._models.values()
        ]
