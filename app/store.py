"""Хранилище: каталог игр, эмбеддинги, факторы ALS, оценки и скрытые игры пользователя."""
import json
import sqlite3
import threading
import time

import numpy as np
import pandas as pd

from .config import (
    ALS_META, ALS_Y_NPY, DB_PATH, EMBEDDINGS_NPY, GAMES_PARQUET, POPULARITY_NPY,
)


class Store:
    def __init__(self) -> None:
        self.games = pd.read_parquet(GAMES_PARQUET)
        self.appids = self.games["appid"].to_numpy(np.int64)
        self.appid_to_idx = {int(a): i for i, a in enumerate(self.appids)}
        self.names = self.games["name"].to_numpy()
        self.years = self.games["year"].to_numpy(np.int32)
        self.images = self.games["header_image"].to_numpy()
        self.descriptions = self.games["description"].to_numpy()
        self.genres = self.games["genres"].to_numpy()
        self.tags = self.games["tags"].to_numpy()
        self.popularity = np.load(POPULARITY_NPY)
        self.embeddings = np.load(EMBEDDINGS_NPY)
        self.Y = np.load(ALS_Y_NPY)
        self.als_meta = json.loads(ALS_META.read_text(encoding="utf-8"))

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ratings(
                game_id INTEGER PRIMARY KEY, rating REAL NOT NULL, ts REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS hidden(
                game_id INTEGER PRIMARY KEY, ts REAL NOT NULL);
            """
        )
        self.lock = threading.Lock()

    # --- оценки (1-10) ---
    def get_ratings(self) -> dict:
        with self.lock:
            rows = self.conn.execute("SELECT game_id, rating FROM ratings").fetchall()
        return {int(g): float(r) for g, r in rows}

    def set_rating(self, appid: int, rating: float) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO ratings(game_id, rating, ts) VALUES(?,?,?) "
                "ON CONFLICT(game_id) DO UPDATE SET rating=excluded.rating, ts=excluded.ts",
                (appid, rating, time.time()),
            )
            self.conn.commit()

    def delete_rating(self, appid: int) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM ratings WHERE game_id=?", (appid,))
            self.conn.commit()

    # --- скрытые из ленты ---
    def get_hidden(self) -> set:
        with self.lock:
            rows = self.conn.execute("SELECT game_id FROM hidden").fetchall()
        return {int(g) for (g,) in rows}

    def hide(self, appid: int) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO hidden(game_id, ts) VALUES(?,?)", (appid, time.time())
            )
            self.conn.commit()

    def unhide(self, appid: int) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM hidden WHERE game_id=?", (appid,))
            self.conn.commit()

    def version(self) -> tuple:
        with self.lock:
            r = self.conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(rating),0), COALESCE(SUM(ts),0) FROM ratings"
            ).fetchone()
            h = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(ts),0) FROM hidden").fetchone()
        return (int(r[0]), float(r[1]), float(r[2]), int(h[0]), float(h[1]))

    # --- поиск по названию ---
    def search(self, query: str, limit: int = 30) -> list:
        q = query.strip().lower()
        if not q:
            return []
        lowered = self.games["name"].str.lower()
        idxs = np.flatnonzero(lowered.str.contains(q, regex=False).to_numpy())
        if idxs.size == 0:
            return []
        names = lowered.to_numpy()[idxs]
        starts = np.array([n.startswith(q) for n in names], dtype=np.float32)
        score = starts * 10.0 + self.popularity[idxs]
        take = min(limit, idxs.size)
        if idxs.size > take:
            local = np.argpartition(-score, take - 1)[:take]
        else:
            local = np.arange(idxs.size)
        top = idxs[local]
        order = np.lexsort((self.names[top].astype(str), -score[local]))
        return [int(i) for i in top[order]]

    def game_json(self, idx: int, rating=None, score=None, reason=None) -> dict:
        item = {
            "game_id": int(self.appids[idx]),
            "name": str(self.names[idx]),
            "year": int(self.years[idx]),
            "image": str(self.images[idx]),
            "genres": [g for g in str(self.genres[idx]).split(";") if g][:6],
            "tags": [t for t in str(self.tags[idx]).split(";") if t][:8],
            "description": str(self.descriptions[idx]),
        }
        if rating is not None:
            item["rating"] = rating
        if score is not None:
            item["score"] = round(float(score), 4)
        if reason:
            item["reason"] = reason
        return item
