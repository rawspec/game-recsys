"""Шаг 1: скачивание сырых датасетов."""
from common import (
    CATALOG_URL,
    RAW_DIR,
    SGR_BASE_URL,
    SGR_DIR,
    SGR_SHARDS,
    download,
)


def main() -> None:
    print("Каталог Steam (~97k игр)...")
    download(CATALOG_URL, RAW_DIR / "steam_games.parquet")

    print("Датасет владений/взаимодействий steamgamerecommender...")
    for shard in SGR_SHARDS:
        for fname in ("users_games.csv", "games.csv"):
            download(f"{SGR_BASE_URL}/{shard}/{fname}", SGR_DIR / shard / fname)

    print("Скачивание завершено.")


if __name__ == "__main__":
    main()
