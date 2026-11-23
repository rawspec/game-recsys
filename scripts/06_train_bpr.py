"""Шаг 6: BPR-MF — байесовское персонализированное ранжирование."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GAMES_PARQUET, INTERACTIONS_PARQUET, MODELS_DIR  # noqa: E402
from trainers import train_bpr  # noqa: E402

FACTORS = 64
EPOCHS = 2


def main() -> None:
    inter = pd.read_parquet(INTERACTIONS_PARQUET)
    n_games = len(pd.read_parquet(GAMES_PARQUET, columns=["appid"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    V = train_bpr(
        inter["user_idx"].to_numpy(), inter["item_idx"].to_numpy(),
        inter["confidence"].to_numpy("float32"), n_games,
        factors=FACTORS, epochs=EPOCHS, verbose=True,
    )
    np.save(MODELS_DIR / "bpr_item_factors.npy", V.astype("float32"))
    (MODELS_DIR / "bpr_meta.json").write_text(
        json.dumps({"factors": FACTORS, "epochs": EPOCHS})
    )
    print(f"BPR готов ({time.time() - t0:.0f} с).", flush=True)
    _exit_no_teardown()


def _exit_no_teardown() -> None:
    # обход сегфолта CUDA при завершении процесса (модель уже сохранена)
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
