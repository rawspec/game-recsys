"""Шаг 2: сборка единого каталога игр и матрицы взаимодействий для ALS."""
import ast
import re

import numpy as np
import pandas as pd

from common import (
    GAMES_PARQUET, INTERACTIONS_PARQUET, RAW_DIR, SGR_DIR, SGR_SHARDS,
)

HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_text(value, limit=1600):
    if not isinstance(value, str) or not value.strip():
        return ""
    text = WS_RE.sub(" ", HTML_RE.sub(" ", value)).strip()
    return text[:limit]


def parse_list_cell(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in value if str(x).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    value = value.strip()
    if value.startswith("["):
        try:
            return [str(x).strip() for x in ast.literal_eval(value) if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [part.strip() for part in value.split(";") if part.strip()]


def dedup_keep_order(items, limit=None):
    seen, out = set(), []
    for item in items:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
        if limit and len(out) >= limit:
            break
    return out


def build_catalog() -> pd.DataFrame:
    print("Загрузка основного каталога (FronkonGames)...")
    raw = pd.read_parquet(RAW_DIR / "steam_games.parquet")
    description = raw["detailed_description"].fillna(raw.get("short_description", ""))
    games = pd.DataFrame({
        "appid": pd.to_numeric(raw["appID"], errors="coerce"),
        "name": raw["name"].fillna("").astype(str).str.strip(),
        "year": pd.to_datetime(raw["release_date"], errors="coerce", format="mixed").dt.year,
        "description": description.map(clean_text),
        "header_image": raw["header_image"].fillna("").astype(str).str.strip(),
        "genres": raw["genres"].map(parse_list_cell),
        "tags": raw["tags"].map(parse_list_cell),
        "categories": raw["categories"].map(parse_list_cell),
        "developers": raw["developers"].map(lambda v: dedup_keep_order(parse_list_cell(v), 4)),
        "price": pd.to_numeric(raw["price"], errors="coerce"),
        "positive": pd.to_numeric(raw["positive"], errors="coerce").fillna(0).astype("int64"),
        "negative": pd.to_numeric(raw["negative"], errors="coerce").fillna(0).astype("int64"),
    })
    games = games.dropna(subset=["appid"]).drop_duplicates(subset="appid", keep="first")
    games["appid"] = games["appid"].astype("int64")
    games["year"] = games["year"].fillna(0).astype("int32")
    games["review_score"] = np.nan
    games["num_reviews"] = np.zeros(len(games), dtype="int64")
    games["num_followers"] = np.zeros(len(games), dtype="int64")
    print(f"  {len(games)} игр")

    print("Слияние со steamgamerecommender...")
    sgr_frames = []
    for shard in SGR_SHARDS:
        path = SGR_DIR / shard / "games.csv"
        if path.exists() and path.stat().st_size > 1024:
            sgr_frames.append(pd.read_csv(path))
    if sgr_frames:
        sgr = pd.concat(sgr_frames, ignore_index=True)
        sgr = sgr.sort_values("numReviews", ascending=False)
        sgr = sgr.drop_duplicates(subset="id", keep="first")
        sgr["genres_list"] = sgr["genres"].map(parse_list_cell)
        sgr["tags_list"] = sgr["tags"].map(parse_list_cell)
        sgr["description_clean"] = sgr["description"].map(clean_text)

        known = sgr[sgr["id"].isin(set(games["appid"]))].set_index("id")
        games["review_score"] = games["appid"].map(known["avgReviewScore"])
        games["num_reviews"] = (
            games["appid"].map(known["numReviews"]).fillna(0).astype("int64")
        )
        games["num_followers"] = (
            games["appid"].map(known["numFollowers"]).fillna(0).astype("int64")
        )
        no_desc = games["description"].eq("")
        games.loc[no_desc, "description"] = (
            games.loc[no_desc, "appid"].map(known["description_clean"]).fillna("")
        )
        games["genres"] = [
            dedup_keep_order(a + (b if isinstance(b, list) else []), limit=10)
            for a, b in zip(games["genres"], games["appid"].map(known["genres_list"]))
        ]
        games["tags"] = [
            dedup_keep_order(a + (b if isinstance(b, list) else []), limit=30)
            for a, b in zip(games["tags"], games["appid"].map(known["tags_list"]))
        ]

        extra = sgr[~sgr["id"].isin(set(games["appid"]))]
        if len(extra):
            rows = pd.DataFrame({
                "appid": extra["id"].astype("int64"),
                "name": extra["name"].fillna("").astype(str).str.strip(),
                "year": np.zeros(len(extra), dtype="int32"),
                "description": extra["description_clean"],
                "header_image": "",
                "genres": extra["genres_list"],
                "tags": extra["tags_list"],
                "categories": [[] for _ in range(len(extra))],
                "developers": [[] for _ in range(len(extra))],
                "price": np.nan,
                "positive": np.zeros(len(extra), dtype="int64"),
                "negative": np.zeros(len(extra), dtype="int64"),
                "review_score": pd.to_numeric(extra["avgReviewScore"], errors="coerce"),
                "num_reviews": pd.to_numeric(extra["numReviews"], errors="coerce").fillna(0).astype("int64"),
                "num_followers": pd.to_numeric(extra["numFollowers"], errors="coerce").fillna(0).astype("int64"),
            })
            games = pd.concat([games, rows], ignore_index=True)
            print(f"  +{len(rows)} игр, которых не было в основном каталоге")

    # обложки по шаблону CDN Steam там, где нет прямой ссылки
    no_img = games["header_image"].eq("")
    games.loc[no_img, "header_image"] = [
        f"https://cdn.cloudflare.steamstatic.com/steam/apps/{a}/header.jpg"
        for a in games.loc[no_img, "appid"]
    ]

    votes = games["positive"] + games["negative"]
    base = np.maximum(votes.to_numpy(), games["num_reviews"].to_numpy())
    popularity = np.log1p(base.astype("float64"))
    if popularity.max() > 0:
        popularity = popularity / popularity.max()
    games["popularity"] = popularity.astype("float32")

    for col in ("genres", "tags", "categories", "developers"):
        games[col] = games[col].map(lambda xs: ";".join(xs))

    games = games.sort_values("popularity", ascending=False, kind="stable")
    games = games.reset_index(drop=True)
    return games


def build_interactions(games: pd.DataFrame) -> None:
    print("Загрузка взаимодействий (может занять пару минут)...")
    frames = []
    for shard in SGR_SHARDS:
        path = SGR_DIR / shard / "users_games.csv"
        frames.append(pd.read_csv(path, usecols=["user_id", "game_id", "playtime_forever"]))
    inter = pd.concat(frames, ignore_index=True)

    inter = inter[inter["playtime_forever"] > 0]
    inter = inter.sort_values(["user_id", "game_id"], kind="stable")
    inter = inter.drop_duplicates(subset=["user_id", "game_id"], keep="last")
    keep_user = inter.groupby("user_id")["game_id"].transform("size") >= 2
    inter = inter[keep_user]

    appid_to_idx = dict(zip(games["appid"].to_numpy(), np.arange(len(games))))
    inter["item_idx"] = inter["game_id"].map(appid_to_idx)
    inter = inter.dropna(subset=["item_idx"])
    inter["item_idx"] = inter["item_idx"].astype("int64")
    inter["user_idx"] = pd.factorize(inter["user_id"])[0].astype("int64")

    hours = inter["playtime_forever"].clip(upper=1000).astype("float32")
    inter["confidence"] = 1.0 + np.log1p(hours)

    out = inter[["user_idx", "item_idx", "confidence"]].reset_index(drop=True)
    out.to_parquet(INTERACTIONS_PARQUET, index=False)
    print(
        f"  {len(out)} взаимодействий, {out['user_idx'].max() + 1} пользователей, "
        f"{out['item_idx'].nunique()} игр с сигналом"
    )


def main() -> None:
    from common import PROC_DIR

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    games = build_catalog()
    build_interactions(games)
    games.to_parquet(GAMES_PARQUET, index=False)
    with_desc = int(games["description"].ne("").sum())
    print(f"Каталог: {len(games)} игр, с описанием: {with_desc}")


if __name__ == "__main__":
    main()
