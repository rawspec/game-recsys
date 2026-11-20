"""Шаг 4: обучение ALS (implicit feedback) на игровом времени."""
import json
import time

import pandas as pd
from common import (
    ALS_FACTORS,
    ALS_FOLD_IN_REG,
    ALS_ITERATIONS,
    ALS_META,
    ALS_REG,
    ALS_Y_NPY,
    GAMES_PARQUET,
    INTERACTIONS_PARQUET,
    POPULARITY_NPY,
)
from trainers import train_als


def main() -> None:
    inter = pd.read_parquet(INTERACTIONS_PARQUET)
    n_games = len(pd.read_parquet(GAMES_PARQUET, columns=["appid"]))
    users = inter["user_idx"].to_numpy()
    items = inter["item_idx"].to_numpy()
    conf = inter["confidence"].to_numpy("float32")
    print(f"ALS: {users.max() + 1} пользователей, {n_games} игр, {len(users)} пар, "
          f"factors={ALS_FACTORS}")
    t0 = time.time()
    Y = train_als(users, items, conf, n_games, factors=ALS_FACTORS,
                  iterations=ALS_ITERATIONS, reg=ALS_REG, verbose=True)
    print(f"  обучение: {time.time() - t0:.0f} с")

    import numpy as np
    np.save(ALS_Y_NPY, Y.astype("float32"))
    pop = pd.read_parquet(GAMES_PARQUET, columns=["popularity"])["popularity"].to_numpy("float32")
    np.save(POPULARITY_NPY, pop)
    ALS_META.write_text(json.dumps({
        "factors": ALS_FACTORS, "reg": ALS_REG, "fold_in_reg": ALS_FOLD_IN_REG,
    }))
    print("ALS готов.")


if __name__ == "__main__":
    main()
