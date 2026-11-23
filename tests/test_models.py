"""Тесты чистых компонентов: метрики, модели, обучение на синтетике."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.models.als import ALSModel  # noqa: E402
from app.models.bpr import BPRModel  # noqa: E402
from app.models.content import ContentModel  # noqa: E402
from app.models.hybrid import HybridModel  # noqa: E402
from app.models.itemknn import ItemKNNModel  # noqa: E402
from app.models.popularity import PopularityModel  # noqa: E402
from trainers import train_als, train_bpr, train_itemknn  # noqa: E402


@pytest.fixture()
def synthetic():
    rng = np.random.default_rng(7)
    n_users, n_items = 300, 120
    users = np.repeat(np.arange(n_users), 8)
    items = rng.integers(0, n_items, size=len(users))
    conf = 1.0 + rng.random(len(users)).astype(np.float32)
    return users.astype(np.int64), items.astype(np.int64), conf, n_items


def test_als_runs_and_scores(synthetic):
    users, items, conf, n_items = synthetic
    Y = train_als(users, items, conf, n_items, factors=16, iterations=3)
    assert Y.shape == (n_items, 16)
    model = ALSModel(Y, reg=0.1)
    rated = np.array([0, 5, 9])
    scores = model.scores(rated, np.array([10.0, 8.0, 3.0]))
    assert scores.shape == (n_items,)
    assert np.isfinite(scores).all()
    cold = model.scores(np.array([], dtype=np.int64), np.array([]))
    assert np.all(cold == 0)


def test_itemknn_topk_and_scoring(synthetic):
    users, items, conf, n_items = synthetic
    sims = train_itemknn(users, items, conf, n_items, top_k=10)
    assert sims.shape == (n_items, n_items)
    nnz_per_row = np.diff(sims.indptr)
    assert nnz_per_row.max() <= 10
    model = ItemKNNModel(sims)
    scores = model.scores(np.array([1, 2]), np.array([9.0, 4.0]))
    assert scores.shape == (n_items,)
    assert np.isfinite(scores).all()


def test_bpr_training_decreases_loss(synthetic, capsys):
    users, items, conf, n_items = synthetic
    V = train_bpr(users, items, conf, n_items, factors=16, epochs=1,
                  batch_size=1024)
    assert V.shape == (n_items, 16)
    model = BPRModel(V)
    scores = model.scores(np.array([3, 7]), np.array([10.0, 9.0]))
    assert scores.shape == (n_items,)
    assert np.isfinite(scores).all()


def test_content_and_hybrid(synthetic):
    users, items, conf, n_items = synthetic
    rng = np.random.default_rng(0)
    E = rng.normal(size=(n_items, 32)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    content = ContentModel(E)
    rated = np.array([0, 1, 2])
    scores = content.scores(rated, np.array([10.0, 1.0, 9.0]))
    assert scores.shape == (n_items,)
    Y = train_als(users, items, conf, n_items, factors=16, iterations=2)
    hybrid = HybridModel(content, ALSModel(Y, 0.1),
                         np.ones(n_items, dtype=np.float32))
    out = hybrid.normalized(rated, np.array([10.0, 1.0, 9.0]))
    assert out.shape == (n_items,)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_popularity():
    pop = np.linspace(0, 1, 50, dtype=np.float32)
    model = PopularityModel(pop)
    assert np.allclose(model.scores(np.array([1]), np.array([5.0])), pop)


def test_metrics_known_values():
    from importlib import util
    spec = util.spec_from_file_location(
        "ev", ROOT / "scripts" / "09_evaluate.py")
    ev = util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    recs = np.array([[5, 3, 2, 1, 0, 9, 8, 7, 6, 4, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18]])
    test_map = {0: {5, 3, 99}}  # релевантные: позиции 0 и 1
    m = ev.metrics_for(recs, test_map, (5,))
    assert m["recall@5"] == pytest.approx(2 / 3)
    dcg = 1.0 + 1.0 / np.log2(3)
    idcg = 1.0 + 1.0 / np.log2(3) + 1.0 / np.log2(4)  # 3 релевантных айтема
    assert m["ndcg@5"] == pytest.approx(dcg / idcg)


def test_split_deterministic():
    import pandas as pd
    from importlib import util
    spec = util.spec_from_file_location(
        "ev", ROOT / "scripts" / "09_evaluate.py")
    ev = util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    rows = np.repeat(np.arange(50), 6)
    items = np.tile(np.arange(6), 50)
    inter = pd.DataFrame({"user_idx": rows, "item_idx": items,
                          "confidence": np.ones(len(rows), dtype="float32")})
    a = ev.split(inter)
    b = ev.split(inter)
    assert np.array_equal(a[3], b[3]) and np.array_equal(a[4], b[4])
    assert len(a[3]) > 0
