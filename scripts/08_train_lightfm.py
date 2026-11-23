"""Шаг 8 (опциональный): LightFM. Пропускается, если пакет не установлен."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GAMES_PARQUET, INTERACTIONS_PARQUET, MODELS_DIR  # noqa: E402

FACTORS = 64
EPOCHS = 3


def main() -> None:
    try:
        from trainers import train_lightfm
    except ImportError:
        print("пакет lightfm не установлен — пропуск (см. README, опционально).")
        return
    inter = pd.read_parquet(INTERACTIONS_PARQUET)
    n_games = len(pd.read_parquet(GAMES_PARQUET, columns=["appid"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    emb, bias = train_lightfm(
        inter["user_idx"].to_numpy(), inter["item_idx"].to_numpy(),
        inter["confidence"].to_numpy("float32"), n_games,
        factors=FACTORS, epochs=EPOCHS, verbose=True,
    )
    np.save(MODELS_DIR / "lightfm_item.npy", emb.astype("float32"))
    np.save(MODELS_DIR / "lightfm_bias.npy", bias.astype("float32"))
    print("LightFM готов.")


if __name__ == "__main__":
    main()
