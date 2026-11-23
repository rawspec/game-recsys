"""Веб-сервер: API + статика."""
import argparse
import json
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .config import WEB_DIR
from .recommender import Recommender
from .store import Store

store = Store()
rec = Recommender(store)
app = FastAPI(title="GameRec", docs_url=None, redoc_url=None)


class RateIn(BaseModel):
    game_id: int
    rating: int = Field(ge=1, le=10)


class HideIn(BaseModel):
    game_id: int


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/stats")
def stats():
    return {
        "catalog": int(len(store.appids)),
        "rated": len(store.get_ratings()),
        "hidden": len(store.get_hidden()),
        "embeddings": int(store.embeddings.shape[1]),
        "als_factors": int(store.Y.shape[1]),
        "algorithms": rec.registry.infos(),
    }


@app.get("/api/algos")
def algos():
    return {"items": rec.registry.infos()}


@app.get("/api/metrics")
def metrics():
    path = Path(__file__).resolve().parents[1] / "data" / "results" / "metrics.json"
    if not path.exists():
        return {"ready": False, "items": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = Path(__file__).resolve().parents[1] / "data" / "results" / "split_meta.json"
    return {"ready": True, "items": data,
            "meta": json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}}


@app.get("/api/search")
def search(q: str = Query(default=""), limit: int = Query(default=30, ge=1, le=100)):
    ratings = store.get_ratings()
    items = [
        store.game_json(i, rating=ratings.get(int(store.appids[i])))
        for i in store.search(q, limit)
    ]
    return {"items": items}


@app.get("/api/recommendations")
def recommendations(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=24, ge=1, le=60),
    algo: str = Query(default="hybrid"),
):
    if algo != "hybrid" and rec.registry.get(algo) is None:
        raise HTTPException(404, f"Алгоритм «{algo}» недоступен (нет артефактов)")
    return rec.page(offset, limit, algo)


@app.get("/api/ratings")
def ratings():
    ratings_map = store.get_ratings()
    idxs = [store.appid_to_idx[a] for a in ratings_map]
    idxs.sort(key=lambda i: (-ratings_map[int(store.appids[i])], str(store.names[i])))
    return {"items": [store.game_json(i, rating=ratings_map[int(store.appids[i])]) for i in idxs]}


@app.post("/api/rate")
def rate(body: RateIn):
    if body.game_id not in store.appid_to_idx:
        raise HTTPException(404, "Игра не найдена в каталоге")
    store.set_rating(body.game_id, float(body.rating))
    return {"ok": True}


@app.delete("/api/rate/{game_id}")
def unrate(game_id: int):
    store.delete_rating(game_id)
    return {"ok": True}


@app.post("/api/hide")
def hide(body: HideIn):
    if body.game_id not in store.appid_to_idx:
        raise HTTPException(404, "Игра не найдена в каталоге")
    store.hide(body.game_id)
    return {"ok": True}


@app.post("/api/unhide")
def unhide(body: HideIn):
    store.unhide(body.game_id)
    return {"ok": True}


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


app.add_middleware(NoCacheMiddleware)

results_dir = Path(__file__).resolve().parents[1] / "data" / "results"
if results_dir.exists():
    app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def _open_when_ready(port: int) -> None:
    import socket

    def waiter() -> None:
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    webbrowser.open(f"http://127.0.0.1:{port}")
                    return
            except OSError:
                import time

                time.sleep(0.5)

    threading.Thread(target=waiter, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="GameRec — рекомендательная система игр")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="сразу открыть в браузере")
    args = parser.parse_args()
    if args.open:
        _open_when_ready(args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
