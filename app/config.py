from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "app.db"
WEB_DIR = ROOT / "web"

GAMES_PARQUET = PROC_DIR / "games.parquet"
EMBEDDINGS_NPY = PROC_DIR / "embeddings.npy"
ALS_Y_NPY = PROC_DIR / "als_item_factors.npy"
POPULARITY_NPY = PROC_DIR / "popularity.npy"
ALS_META = PROC_DIR / "als_meta.json"
