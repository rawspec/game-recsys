"""Шаг 9: офлайн-оценка алгоритмов на holdout-сплите.

Сплит: у каждого пользователя 80% взаимодействий — train, 20% — test
(только пользователи с >=5 взаимодействиями). Релевантность = игра есть в тесте.
Метрики: Recall@K, NDCG@K, MAP@K (K=10, 20), Coverage@20, Novelty, Diversity.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ALS_FOLD_IN_REG, ALS_REG, EMBEDDINGS_NPY, GAMES_PARQUET, INTERACTIONS_PARQUET,
    MODELS_DIR, POPULARITY_NPY, RESULTS_DIR,
)
from trainers import (  # noqa: E402
    train_als, train_bpr, train_itemknn, train_twotower,
)

SEED = 42
MIN_INTERACTIONS = 5
TEST_FRAC = 0.2
MAX_TEST_USERS = 8000
EVAL_RATING = 8.0  # «оценка», которой заменяем implicit-сигнал при профиле
K_S = (10, 20)


def split(inter: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    users = inter["user_idx"].to_numpy()
    items = inter["item_idx"].to_numpy()
    conf = inter["confidence"].to_numpy("float32")
    test_mask = np.zeros(len(inter), dtype=bool)
    order = np.argsort(users, kind="stable")
    u_s = users[order]
    boundaries = np.flatnonzero(np.diff(u_s))
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries + 1, [len(u_s)]))
    test_users_all = []
    for s, e in zip(starts, ends):
        n = e - s
        if n >= MIN_INTERACTIONS:
            k = max(1, int(round(n * TEST_FRAC)))
            choice = rng.choice(n, size=k, replace=False)
            test_mask[order[s + choice]] = True
            test_users_all.append(u_s[s])
    if len(test_users_all) > MAX_TEST_USERS:
        keep = rng.choice(np.array(test_users_all), MAX_TEST_USERS, replace=False)
        keep_set = set(keep.tolist())
        test_mask &= np.isin(users, list(keep_set))
        test_users_all = keep.tolist()
    train_users = users[~test_mask]
    train_items = items[~test_mask]
    train_conf = conf[~test_mask]
    return (train_users, train_items, train_conf,
            users[test_mask], items[test_mask], np.array(sorted(test_users_all)))


def csr_profile(user_ids, items, n_items, value=EVAL_RATING):
    ones = np.ones(len(user_ids), dtype=np.float32)
    return sparse.csr_matrix((ones, (user_ids, items)), shape=(user_ids.max() + 1, n_items))


def _dense(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def batch_scores(model_name, chunk_users, profile_chunk, ctx):
    """profile_chunk — CSR (B x n_items) с единичными весами."""
    n_items = ctx["n_items"]
    if model_name == "popularity":
        return np.tile(ctx["popularity"], (len(chunk_users), 1))
    if model_name == "content":
        E = ctx["E"]
        u = _dense(profile_chunk @ E)
        u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
        return _dense(u @ E.T)
    if model_name == "als":
        return foldin_als(profile_chunk, ctx["Y"], ctx["reg"])
    if model_name == "itemknn":
        p = profile_chunk.multiply(1.0 / (profile_chunk.sum(1) + 1e-9))
        return _dense(p @ ctx["sims"])
    if model_name == "bpr":
        V = ctx["V_bpr"]
        p = profile_chunk.multiply(1.0 / (profile_chunk.sum(1) + 1e-9))
        u = _dense(p @ V)
        u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
        return _dense(u @ V.T)
    if model_name == "twotower":
        E, w1, b1, w2, b2 = ctx["tt"]
        p = profile_chunk.multiply(1.0 / (profile_chunk.sum(1) + 1e-9))
        z = _dense(p @ E)
        h = np.maximum(z @ w1 + b1, 0.0)
        u = h @ w2 + b2
        u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
        return _dense(u @ E.T)
    if model_name == "lightfm":
        V, b = ctx["lfm"]
        p = profile_chunk.multiply(1.0 / (profile_chunk.sum(1) + 1e-9))
        return _dense(p @ V.T + b[None, :])
    raise ValueError(model_name)


def foldin_als(profile_chunk, Y, reg):
    B = profile_chunk.shape[0]
    out = np.zeros((B, Y.shape[0]), dtype=np.float32)
    indptr, indices = profile_chunk.indptr, profile_chunk.indices
    eye = np.eye(Y.shape[1], dtype=np.float32)
    for b in range(B):
        s, e = indptr[b], indptr[b + 1]
        if e == s:
            continue
        G = Y[indices[s:e]]
        c = np.full(e - s, EVAL_RATING, dtype=np.float32)
        A = (G.T * c) @ G + reg * eye
        x = np.linalg.solve(A, G.T @ c)
        out[b] = Y @ x
    return out


def metrics_for(recs: np.ndarray, test_map: dict, ks):
    out = {}
    n = recs.shape[0]
    for k in ks:
        hits = np.zeros(n)
        dcg = np.zeros(n)
        ap = np.zeros(n)
        for i in range(n):
            rel = test_map[i]
            if not rel:
                continue
            top = recs[i, :k]
            hit_positions = np.nonzero(np.isin(top, list(rel)))[0]
            hits[i] = len(hit_positions) / len(rel)
            if len(hit_positions):
                dcg[i] = float((1.0 / np.log2(hit_positions + 2.0)).sum())
            # AP
            prec = 0.0
            for rank, pos in enumerate(hit_positions):
                prec += (rank + 1) / (pos + 1)
            ap[i] = prec / min(len(rel), k)
        idcg = np.array([sum(1.0 / np.log2(r + 2) for r in range(min(len(test_map[i]), k))) for i in range(n)])
        ndcg = np.divide(dcg, idcg, out=np.zeros(n), where=idcg > 0)
        out[f"recall@{k}"] = float(hits.mean())
        out[f"ndcg@{k}"] = float(ndcg.mean())
        out[f"map@{k}"] = float(ap.mean())
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inter = pd.read_parquet(INTERACTIONS_PARQUET)
    n_games = len(pd.read_parquet(GAMES_PARQUET, columns=["appid"]))
    print(f"Взаимодействий: {len(inter)}, игр: {n_games}")

    (tr_u, tr_i, tr_c, te_u, te_i, test_users) = split(inter)
    print(f"Train: {len(tr_u)} пар | Test: {len(te_u)} пар, {len(test_users)} пользователей")

    # популярность по train
    counts = np.bincount(tr_i, minlength=n_games).astype("float64")
    pop = np.log1p(counts)
    pop = (pop / pop.max()).astype("float32")

    ctx = {"n_items": n_games, "popularity": pop, "E": np.load(EMBEDDINGS_NPY)}

    print("Обучаю модели на train-сплите...")
    interacted = np.zeros(n_games, dtype=bool)
    interacted[np.unique(tr_i)] = True
    non_interacted = ~interacted

    cache = RESULTS_DIR / "eval_cache"
    meta_file = cache / "meta.json"
    cache_valid = meta_file.exists() and json.loads(meta_file.read_text()) == {
        "seed": SEED, "test_frac": TEST_FRAC, "min_interactions": MIN_INTERACTIONS,
        "train_pairs": int(len(tr_u)), "als_reg": ALS_REG,
    }

    if cache_valid:
        print("  кэш обученных моделей найден — загружаю")
        ctx["Y"] = np.load(cache / "als_Y.npy")
        ctx["reg"] = ALS_FOLD_IN_REG
        ctx["sims"] = sparse.load_npz(cache / "itemknn.npz")
        ctx["V_bpr"] = np.load(cache / "bpr_V.npy")
        ctx["tt"] = tuple(np.load(cache / f"tt_{k}.npy") for k in ("item", "w1", "b1", "w2", "b2"))
    else:
        t0 = time.time()
        ctx["Y"] = train_als(tr_u, tr_i, tr_c, n_games, iterations=12, reg=ALS_REG,
                             verbose=True)
        ctx["reg"] = ALS_FOLD_IN_REG
        print(f"  ALS: {time.time() - t0:.0f} с", flush=True)
        t0 = time.time()
        ctx["sims"] = train_itemknn(tr_u, tr_i, tr_c, n_games, top_k=200, verbose=True)
        print(f"  ItemKNN: {time.time() - t0:.0f} с", flush=True)
        t0 = time.time()
        ctx["V_bpr"] = train_bpr(tr_u, tr_i, tr_c, n_games, epochs=2, verbose=True)
        print(f"  BPR: {time.time() - t0:.0f} с", flush=True)
        t0 = time.time()
        ctx["tt"] = train_twotower(tr_u, tr_i, tr_c, n_games, epochs=1, verbose=True)
        print(f"  Two-Tower: {time.time() - t0:.0f} с", flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "als_Y.npy", ctx["Y"])
        sparse.save_npz(cache / "itemknn.npz", ctx["sims"])
        np.save(cache / "bpr_V.npy", ctx["V_bpr"])
        for k, v in zip(("item", "w1", "b1", "w2", "b2"), ctx["tt"]):
            np.save(cache / f"tt_{k}.npy", v)
        meta_file.write_text(json.dumps({
            "seed": SEED, "test_frac": TEST_FRAC, "min_interactions": MIN_INTERACTIONS,
            "train_pairs": int(len(tr_u)), "als_reg": ALS_REG,
        }))

    models = ["popularity", "content", "als", "itemknn", "bpr", "twotower"]
    try:
        from trainers import train_lightfm
        ctx["lfm"] = train_lightfm(tr_u, tr_i, tr_c, n_games, epochs=3, verbose=True)
        models.append("lightfm")
    except ImportError:
        print("  LightFM пропущен (пакет не установлен)")

    # train-профили тестовых пользователей (для исключения уже виденного)
    train_csr = csr_profile(tr_u, tr_i, n_games)
    test_map = {}
    order = np.argsort(te_u, kind="stable")
    te_u_s, te_i_s = te_u[order], te_i[order]
    boundaries = np.flatnonzero(np.diff(te_u_s))
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries + 1, [len(te_u_s)]))
    pos = {u: i for i, u in enumerate(test_users)}
    for s, e in zip(starts, ends):
        if te_u_s[s] in pos:
            test_map[pos[te_u_s[s]]] = set(te_i_s[s:e].tolist())

    # сэмпл рекомендаций для coverage/novelty/diversity
    sample_n = min(2000, len(test_users))

    results = {}
    CHUNK = 512
    for name in models:
        t0 = time.time()
        recalls20 = []
        all_recs = []
        for c0 in range(0, len(test_users), CHUNK):
            chunk = test_users[c0:c0 + CHUNK]
            profile = train_csr[chunk]
            scores = batch_scores(name, chunk, profile, ctx)
            indptr = profile.indptr
            for b in range(len(chunk)):
                s, e = indptr[b], indptr[b + 1]
                if e > s:
                    scores[b, profile.indices[s:e]] = -np.inf
            scores[:, non_interacted] = -1e9
            top = np.argpartition(-scores, K_S[1], axis=1)[:, :K_S[1]]
            top = top[np.arange(len(chunk))[:, None], np.argsort(-np.take_along_axis(scores, top, 1), axis=1)]
            if c0 < sample_n:
                all_recs.append(top[:, :20])
            recalls20.append(top)
        recs = np.vstack(recalls20)
        m = metrics_for(recs, test_map, K_S)

        sample_recs = np.vstack(all_recs) if all_recs else recs[:sample_n, :20]
        uniq = np.unique(sample_recs)
        m["coverage@20"] = float(len(uniq) / max(1, len(np.unique(tr_i))))
        pop_norm = np.maximum(pop, 1e-9)
        m["novelty"] = float(np.mean(-np.log2(pop_norm[sample_recs])))
        E = ctx["E"]
        sub = E[sample_recs[:, :10]]
        sims = np.einsum("bnd,bmd->bnm", sub, sub)
        m["diversity"] = float(1.0 - np.mean(sims))
        results[name] = m
        print(f"  {name}: " + " | ".join(f"{k}={v:.4f}" for k, v in m.items()),
              flush=True)
        print(f"    ({time.time() - t0:.0f} с)", flush=True)

    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    write_markdown(results)
    plot(results)
    print("Готово: data/results/metrics.{json,md,png}")


def write_markdown(results):
    order = ["popularity", "content", "itemknn", "als", "bpr", "twotower", "lightfm"]
    keys = ["recall@10", "recall@20", "ndcg@10", "ndcg@20", "map@10",
            "coverage@20", "novelty", "diversity"]
    lines = ["| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MAP@10 | Coverage@20 | Novelty | Diversity |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        cells = " | ".join(f"{r[k]:.4f}" for k in keys)
        lines.append(f"| {name} | {cells} |")
    (RESULTS_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, ["recall@10", "ndcg@10", "coverage@20"]):
        vals = [results[n][metric] for n in names]
        ax.bar(names, vals)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
