"""Шаг 3: эмбеддинги контента игр ( sentence-transformers, GPU если есть)."""
import os

# обход сегфолта в transformers 5.22 при многопоточной материализации весов
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import numpy as np
import pandas as pd
import torch
from common import EMBED_MODEL, EMBEDDINGS_NPY, GAMES_PARQUET
from sentence_transformers import SentenceTransformer


def make_text(row) -> str:
    parts = [row.name]
    if row.genres:
        parts.append("Genres: " + row.genres.replace(";", ", "))
    if row.tags:
        parts.append("Tags: " + "; ".join(row.tags.split(";")[:15]))
    if row.description:
        parts.append(row.description)
    return ". ".join(parts)


def main() -> None:
    games = pd.read_parquet(GAMES_PARQUET)
    texts = [make_text(row) for row in games.itertuples(index=False)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Модель: {EMBED_MODEL} | устройство: {device} | текстов: {len(texts)}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    emb = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save(EMBEDDINGS_NPY, emb.astype("float32"))
    print(f"Сохранено: {EMBEDDINGS_NPY} {emb.shape}")


if __name__ == "__main__":
    main()
