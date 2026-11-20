"""Шаг 5: ItemKNN — косинусная item-item схожесть по взаимодействиям."""
import sys
from pathlib import Path

import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GAMES_PARQUET, INTERACTIONS_PARQUET, MODELS_DIR
from trainers import train_itemknn


def main() -> None:
    inter = pd.read_parquet(INTERACTIONS_PARQUET)
    n_games = len(pd.read_parquet(GAMES_PARQUET, columns=["appid"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sims = train_itemknn(
        inter["user_idx"].to_numpy(), inter["item_idx"].to_numpy(),
        inter["confidence"].to_numpy("float32"), n_games, top_k=200, verbose=True,
    )
    sparse.save_npz(MODELS_DIR / "itemknn_sims.npz", sims)
    print(f"ItemKNN готов: {sims.nnz} ненулевых схожестей.")


if __name__ == "__main__":
    main()
