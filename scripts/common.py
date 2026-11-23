"""Общие константы и helpers для пайплайна."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
SGR_DIR = RAW_DIR / "sgr"

GAMES_PARQUET = PROC_DIR / "games.parquet"
INTERACTIONS_PARQUET = PROC_DIR / "interactions.parquet"
EMBEDDINGS_NPY = PROC_DIR / "embeddings.npy"
ALS_Y_NPY = PROC_DIR / "als_item_factors.npy"
POPULARITY_NPY = PROC_DIR / "popularity.npy"
ALS_META = PROC_DIR / "als_meta.json"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

CATALOG_URL = "https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/main/data/train-00000-of-00001.parquet"
SGR_BASE_URL = "https://huggingface.co/datasets/steamgamerecommender/data_files_public/resolve/main"
SGR_SHARDS = ["0", "20000", "39935", "59526"]

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ALS_FACTORS = 64
ALS_ITERATIONS = 12
# сильная регуляризация подобрана офлайн-оценкой: пользователи в среднем имеют
# ~170 игр, «учебный» reg=0.1 даёт recall@10=0.001; reg=32 → 0.084 (см. README)
ALS_REG = 32.0
ALS_FOLD_IN_REG = 100.0  # регуляризация fold-in нового пользователя


def download(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  пропуск (уже скачан): {dest.name} ({dest.stat().st_size / 1e6:.1f} МБ)")
        return
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {got / 1e6:.0f}/{total / 1e6:.0f} МБ", end="", flush=True)
    print()
    tmp.replace(dest)
