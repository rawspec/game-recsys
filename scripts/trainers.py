"""Ядро обучения всех моделей. Используется скриптами 04-08 и офлайн-оценкой."""
from __future__ import annotations

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------- ALS (implicit)
def _solve_side(order, a, b, c, W, other, reg, eye):
    a_s, b_s, c_s = a[order], b[order], c[order]
    boundaries = np.flatnonzero(np.diff(a_s))
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries + 1, [len(a_s)]))
    for s, e in zip(starts, ends):
        G = other[b_s[s:e]]
        cw = c_s[s:e]
        A = (G.T * cw) @ G + reg * eye
        bvec = G.T @ cw
        W[a_s[s]] = np.linalg.solve(A, bvec)


def train_als(users, items, conf, n_items, factors=64, iterations=12, reg=0.1,
              seed=42, verbose=False):
    """Классический ALS из Hu et al. (2008), p=1, confidence из conf."""
    n_users = int(users.max()) + 1
    rng = np.random.default_rng(seed)
    X = np.zeros((n_users, factors), dtype=np.float32)
    Y = rng.normal(0.0, 0.05, size=(n_items, factors)).astype(np.float32)
    eye = np.eye(factors, dtype=np.float32)
    user_order = np.argsort(users, kind="stable")
    item_order = np.argsort(items, kind="stable")
    for it in range(1, iterations + 1):
        _solve_side(user_order, users, items, conf, X, Y, reg, eye)
        _solve_side(item_order, items, users, conf, Y, X, reg, eye)
        if verbose:
            print(f"  ALS итерация {it}/{iterations}", flush=True)
    interacted = np.zeros(n_items, dtype=bool)
    interacted[np.unique(items)] = True
    Y[~interacted] = 0.0
    return Y


# ---------------------------------------------------------------- ItemKNN
def build_user_item(users, items, conf, n_users, n_items) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (conf, (users, items)), shape=(n_users, n_items), dtype=np.float32
    )


def train_itemknn(users, items, conf, n_items, top_k=200, verbose=False):
    n_users = int(users.max()) + 1
    user_item = build_user_item(users, items, np.ones_like(conf), n_users, n_items)
    item_user = user_item.T.tocsr()
    if verbose:
        print("  ItemKNN: считаю косинусные схожести...", flush=True)
    norms = np.sqrt(item_user.power(2).sum(axis=1)).A.ravel() + 1e-12
    normed = sparse.diags(1.0 / norms) @ item_user
    sims = (normed @ normed.T).tocsr()
    sims.setdiag(0.0)
    sims.eliminate_zeros()
    if verbose:
        print(f"  ItemKNN: nnz={sims.nnz}, оставляю top-{top_k}...", flush=True)
    indptr, indices, data = [0], [], []
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz > top_k:
            part = np.argpartition(-row.data, top_k)[:top_k]
            idx = row.indices[part]
            vals = row.data[part]
            order = np.argsort(-vals)
            idx, vals = idx[order], vals[order]
        else:
            idx, vals = row.indices, row.data
        indices.extend(idx.tolist())
        data.extend(vals.tolist())
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
        shape=sims.shape,
        dtype=np.float32,
    )


# ---------------------------------------------------------------- BPR-MF
def train_bpr(users, items, conf, n_items, factors=64, epochs=2, lr=0.05,
              reg=1e-5, batch_size=8192, seed=42, verbose=False):
    """BPR с user-представлением = среднее эмбеддингов истории (как на инференсе).

    Реализация на torch (GPU если есть): автоградиент + Adagrad. Быстро.
    Негативы сэмплируются равномерно по interacted-айтемам.
    """
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_users = int(users.max()) + 1
    ui = build_user_item(users, items, np.ones_like(conf), n_users, n_items).tocoo()
    hist = torch.sparse_coo_tensor(
        np.vstack([ui.row, ui.col]),
        torch.ones(len(ui.row), dtype=torch.float32),
        ui.shape, device=device,
    ).coalesce()

    V = (torch.randn(n_items, factors, device=device) * 0.05).requires_grad_(True)
    accum = torch.zeros(n_items, factors, device=device)
    interacted = torch.tensor(np.unique(items), device=device)
    users_t = torch.tensor(users, device=device)
    items_t = torch.tensor(items, device=device)
    ones_col = torch.ones(n_items, 1, device=device)
    inv_cnt = 1.0 / (torch.sparse.mm(hist, ones_col).squeeze(1) + 1e-9)

    n_pairs = len(users)
    for epoch in range(epochs):
        perm = rng.permutation(n_pairs)
        total, seen = 0.0, 0
        for s in range(0, n_pairs, batch_size):
            idx = torch.as_tensor(perm[s:s + batch_size], device=device)
            bu, bi = users_t[idx], items_t[idx]
            bj = interacted[torch.randint(0, len(interacted), (len(idx),), device=device)]

            M = torch.sparse.mm(hist, V)[bu] * inv_cnt[bu][:, None]
            diff = (M * (V[bi] - V[bj])).sum(dim=1)
            loss = F.softplus(-diff).sum()
            grad = torch.autograd.grad(loss, V)[0]

            with torch.no_grad():
                accum.addcmul_(grad, grad)
                V.sub_(lr * grad / (accum.sqrt() + 1e-8))
            total += float(loss)
            seen += len(idx)
        if verbose:
            print(f"  BPR эпоха {epoch + 1}/{epochs} — loss {total / seen:.4f}",
                  flush=True)

    with torch.no_grad():
        out = V.detach().clone()
        mask = torch.ones(n_items, dtype=torch.bool, device=device)
        mask[interacted] = False
        out[mask] = 0.0
    return out.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- Two-Tower
def train_twotower(users, items, conf, n_items, factors=128, epochs=1,
                   batch_size=16384, lr=1e-3, tau=0.07, seed=42, verbose=False):
    """Two-Tower с InfoNCE (in-batch negatives) и поправкой на частоту.

    User-tower: LayerNorm(Linear(ReLU(Linear(среднее эмбеддингов истории)))).
    Возвращает (item_emb, w1, b1, w2, b2) — всё numpy для инференса без torch.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    n_users = int(users.max()) + 1

    item_emb = nn.Embedding(n_items, factors).to(device)
    nn.init.normal_(item_emb.weight, std=0.05)
    mlp = nn.Sequential(
        nn.Linear(factors, factors), nn.ReLU(), nn.Linear(factors, factors)
    ).to(device)
    norm = nn.LayerNorm(factors).to(device)

    # разреженная история (users x items) для user-tower
    ui = build_user_item(users, items, conf.astype(np.float32), n_users, n_items)
    ui_coo = ui.tocoo()
    hist = torch.sparse_coo_tensor(
        np.vstack([ui_coo.row, ui_coo.col]), torch.tensor(ui_coo.data),
        ui.shape, device=device,
    ).coalesce()

    counts = np.bincount(items, minlength=n_items).astype(np.float64)
    log_q = torch.tensor(
        np.log(np.maximum(counts, 1) ** 0.75), dtype=torch.float32, device=device
    )

    params = list(item_emb.parameters()) + list(mlp.parameters()) + list(norm.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    n_pairs = len(users)
    for epoch in range(epochs):
        perm = np.random.permutation(n_pairs)
        total, seen = 0.0, 0
        for s in range(0, n_pairs, batch_size):
            idx = perm[s:s + batch_size]
            bu = torch.tensor(users[idx], device=device)
            bi = torch.tensor(items[idx], device=device)

            V = item_emb.weight
            h_u = torch.sparse.mm(hist, V)[bu]        # (B, f) взвешенное среднее
            h_u = norm(mlp(h_u))
            h_u = F.normalize(h_u, dim=-1)
            v_i = F.normalize(item_emb(bi), dim=-1)

            pos = (h_u * v_i).sum(-1) / tau - log_q[bi]
            neg = (h_u @ v_i.T) / tau - log_q[bi][None, :]
            logits = torch.cat([pos[:, None], neg], dim=1)
            loss = F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long,
                                                       device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            seen += len(idx)
        if verbose:
            print(f"  Two-Tower эпоха {epoch + 1}/{epochs} — loss {total / seen:.4f}",
                  flush=True)

    E = F.normalize(item_emb.weight.detach(), dim=-1).cpu().numpy()
    w1 = mlp[0].weight.detach().T.cpu().numpy()   # (f, f) под numpy z @ w + b
    b1 = mlp[0].bias.detach().cpu().numpy()
    w2 = mlp[2].weight.detach().T.cpu().numpy()
    b2 = mlp[2].bias.detach().cpu().numpy()
    interacted = np.unique(items)
    E[~np.isin(np.arange(n_items), interacted)] = 0.0
    return E.astype(np.float32), w1, b1, w2, b2


# ---------------------------------------------------------------- LightFM (optional)
def train_lightfm(users, items, conf, n_items, factors=64, epochs=3, seed=42,
                  verbose=False, max_users=40000):
    """Гибридный LightFM (logistic). Требует пакет lightfm; иначе ImportError."""
    from lightfm import LightFM

    n_users_all = int(users.max()) + 1
    if n_users_all > max_users:  # ограничение по скорости
        keep = np.random.default_rng(seed).choice(
            n_users_all, size=max_users, replace=False
        )
        mask = np.isin(users, keep)
        remap = {u: i for i, u in enumerate(keep)}
        users = np.array([remap[u] for u in users[mask]], dtype=np.int64)
        items = items[mask]
        conf = conf[mask]
        n_users = max_users
    else:
        n_users = n_users_all
    ui = build_user_item(users, items, conf, n_users, n_items)
    model = LightFM(no_components=factors, loss="logistic", random_state=seed)
    for ep in range(epochs):
        model.fit(ui, epochs=1, num_threads=0, verbose=verbose)
        if verbose:
            print(f"  LightFM эпоха {ep + 1}/{epochs}", flush=True)
    return model.item_embeddings.astype(np.float32), model.item_biases.astype(np.float32)
