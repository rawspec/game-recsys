"""Шаг 7: Two-Tower нейросеть (PyTorch, GPU) с InfoNCE."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GAMES_PARQUET, INTERACTIONS_PARQUET, MODELS_DIR  # noqa: E402
from trainers import train_twotower  # noqa: E402

FACTORS = 128
EPOCHS = 1


def main() -> None:
    inter = pd.read_parquet(INTERACTIONS_PARQUET)
    n_games = len(pd.read_parquet(GAMES_PARQUET, columns=["appid"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    E, w1, b1, w2, b2 = train_twotower(
        inter["user_idx"].to_numpy(), inter["item_idx"].to_numpy(),
        inter["confidence"].to_numpy("float32"), n_games,
        factors=FACTORS, epochs=EPOCHS, verbose=True,
    )
    np.save(MODELS_DIR / "twotower_item.npy", E)
    np.save(MODELS_DIR / "twotower_w1.npy", w1)
    np.save(MODELS_DIR / "twotower_b1.npy", b1)
    np.save(MODELS_DIR / "twotower_w2.npy", w2)
    np.save(MODELS_DIR / "twotower_b2.npy", b2)
    (MODELS_DIR / "twotower_meta.json").write_text(
        json.dumps({"factors": FACTORS, "epochs": EPOCHS, "loss": "infonce"})
    )
    print(f"Two-Tower готов ({time.time() - t0:.0f} с).", flush=True)
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
