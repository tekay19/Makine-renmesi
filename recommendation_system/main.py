import os
import math
import random
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from joblib import dump, load

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DATA_DIR = Path("data")
INTERACTIONS_PATH = DATA_DIR / "interactions.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

POPULARITY_PATH = ARTIFACT_DIR / "pop.npy"
ITEM_SIMS_PATH = ARTIFACT_DIR / "item_sims.npy"
USER_MAP_PATH = ARTIFACT_DIR / "user_map.joblib"
ITEM_MAP_PATH = ARTIFACT_DIR / "item_map.joblib"
TFIDF_MTX_PATH = ARTIFACT_DIR / "tfidf.npy"
TFIDF_VECT_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"

# ---------------------------
# 1) Veri yardımcıları
# ---------------------------

def ensure_demo_data():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    if not PRODUCTS_PATH.exists():
        rng = random.Random(42)
        brands = ["Acme", "Zen", "Nova", "Peak"]
        cats = ["Shoe", "Tshirt", "PhoneCase", "Headphone", "Backpack"]
        rows = []
        for pid in range(1, 201):
            brand = rng.choice(brands)
            cat = rng.choice(cats)
            tags = " ".join(rng.sample(
                ["sport", "leather", "casual", "running", "wireless", "noise-cancel", "slim", "classic",
                 "travel", "office", "gaming", "summer", "winter"], k=3))
            title = f"{brand} {cat} {pid}"
            price = round(rng.uniform(9.9, 299.9), 2)
            rows.append([pid, title, brand, cat, tags, price])
        dfp = pd.DataFrame(rows, columns=["product_id", "title", "brand", "category", "tags", "price"])
        dfp.to_csv(PRODUCTS_PATH, index=False)

    if not INTERACTIONS_PATH.exists():
        rng = random.Random(123)
        users = list(range(1, 401))
        prods = pd.read_csv(PRODUCTS_PATH)["product_id"].tolist()
        events = ["view", "cart", "purchase"]
        weights = {"view": 1.0, "cart": 3.0, "purchase": 5.0}
        rows = []
        # Sentetik etkileşim: her kullanıcı ~80 kayıt
        for u in users:
            for _ in range(80):
                p = rng.choice(prods)
                e = random.choices(events, weights=[0.7, 0.2, 0.1], k=1)[0]
                ts = rng.randint(1_700_000_000, 1_750_000_000)
                rows.append([u, p, e, ts])
        dfi = pd.DataFrame(rows, columns=["user_id", "product_id", "event", "ts"])
        dfi.to_csv(INTERACTIONS_PATH, index=False)


def load_data():
    ensure_demo_data()
    products = pd.read_csv(PRODUCTS_PATH)
    interactions = pd.read_csv(INTERACTIONS_PATH)
    # Temizlik
    products = products.drop_duplicates("product_id").reset_index(drop=True)
    interactions = interactions.dropna(subset=["user_id", "product_id"])
    # cast
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["product_id"] = interactions["product_id"].astype(int)
    return products, interactions


# ---------------------------
# 2) Önericiler (Popularity, ItemCF, Content)
# ---------------------------

def build_mappings(items: pd.Series, users: pd.Series):
    item_ids = np.sort(items.unique())
    user_ids = np.sort(users.unique())
    item2idx = {pid: i for i, pid in enumerate(item_ids)}
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    idx2item = {i: pid for pid, i in item2idx.items()}
    idx2user = {i: uid for uid, i in user2idx.items()}
    return item2idx, idx2item, user2idx, idx2user

def event_weight(e: str) -> float:
    # Etkinlik ağırlıkları (kolay ayar)
    return {"view": 1.0, "cart": 3.0, "purchase": 5.0}.get(e, 0.5)

def user_item_matrix(interactions: pd.DataFrame, item2idx: Dict[int,int], user2idx: Dict[int,int]):
    # Basit yoğun matris (küçük-orta veri için yeterli). Büyük veri için sparse matris kullan.
    n_users = len(user2idx)
    n_items = len(item2idx)
    UI = np.zeros((n_users, n_items), dtype=np.float32)
    for _, row in interactions.iterrows():
        u = user2idx.get(row["user_id"])
        i = item2idx.get(row["product_id"])
        if u is None or i is None:
            continue
        UI[u, i] += event_weight(str(row["event"]))
    return UI

def train_popularity(interactions: pd.DataFrame, item2idx: Dict[int,int]) -> np.ndarray:
    pop = np.zeros(len(item2idx), dtype=np.float32)
    for _, row in interactions.iterrows():
        i = item2idx.get(row["product_id"])
        if i is None: 
            continue
        pop[i] += event_weight(str(row["event"]))
    return pop

def train_itemcf(UI: np.ndarray) -> np.ndarray:
    # Kosinüs benzerliği (item x item)
    # Normalize columns (items) then dot
    I = normalize(UI, axis=0)  # kullanıcı ekseninde L2 normalize
    sims = I.T @ I              # (items x items)
    np.fill_diagonal(sims, 0.0) # kendisiyle benzerliği 0 yap
    return sims.astype(np.float32)

def train_content(products: pd.DataFrame):
    # Basit içerik temsili: title + brand + category + tags
    text = (products["title"].fillna("") + " " +
            products["brand"].fillna("") + " " +
            products["category"].fillna("") + " " +
            products["tags"].fillna("")).values
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vectorizer.fit_transform(text)
    return X, vectorizer

def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[-1])
    if k <= 0:
        return np.array([], dtype=int)
    # argpartition ile O(n) seçme
    idx = np.argpartition(-scores, k-1)[:k]
    # skor sırasına göre diz
    return idx[np.argsort(-scores[idx])]

# ---------------------------
# 3) Eğitim/Artifakt Kaydetme
# ---------------------------

def train_and_persist():
    products, interactions = load_data()
    item2idx, idx2item, user2idx, idx2user = build_mappings(
        products["product_id"], interactions["user_id"]
    )
    UI = user_item_matrix(interactions, item2idx, user2idx)

    # Popülerlik
    popularity = train_popularity(interactions, item2idx)
    np.save(POPULARITY_PATH, popularity)

    # Item-Item CF
    item_sims = train_itemcf(UI)
    np.save(ITEM_SIMS_PATH, item_sims)

    # Content (TF-IDF)
    tfidf_mtx, vectorizer = train_content(products)
    np.save(TFIDF_MTX_PATH, tfidf_mtx.toarray().astype(np.float32))
    dump(vectorizer, TFIDF_VECT_PATH)

    # Haritalar
    dump({"item2idx": item2idx, "idx2item": idx2item}, ITEM_MAP_PATH)
    dump({"user2idx": user2idx, "idx2user": idx2user}, USER_MAP_PATH)

    return {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "ui_shape": list(UI.shape),
        "products": len(products),
        "interactions": len(interactions)
    }

# ---------------------------
# 4) Öneri Fonksiyonları (Serving)
# ---------------------------

class RecResponse(BaseModel):
    user_id: Optional[int] = None
    product_id: Optional[int] = None
    recs: List[Dict[str, object]]

def load_artifacts():
    products = pd.read_csv(PRODUCTS_PATH)
    pop = np.load(POPULARITY_PATH, allow_pickle=False)
    sims = np.load(ITEM_SIMS_PATH, allow_pickle=False)
    tfidf = np.load(TFIDF_MTX_PATH, allow_pickle=False)
    vect = load(TFIDF_VECT_PATH)
    item_maps = load(ITEM_MAP_PATH)
    user_maps = load(USER_MAP_PATH)
    return products, pop, sims, tfidf, vect, item_maps, user_maps

def recommend_user(user_id: int, k: int = 10) -> List[int]:
    products, pop, sims, tfidf, vect, item_maps, user_maps = load_artifacts()
    item2idx, idx2item = item_maps["item2idx"], item_maps["idx2item"]
    user2idx = user_maps["user2idx"]

    if user_id not in user2idx:
        # soğuk başlangıç: en popüler
        top_idx = topk_from_scores(pop, k)
        return [idx2item[int(i)] for i in top_idx]

    # Kullanıcının geçmişini al
    inter = pd.read_csv(INTERACTIONS_PATH)
    hist = inter.loc[inter["user_id"] == user_id, "product_id"].tolist()
    hist_idx = [item2idx[p] for p in hist if p in item2idx]
    if len(hist_idx) == 0:
        top_idx = topk_from_scores(pop, k)
        return [idx2item[int(i)] for i in top_idx]

    # ItemCF: geçmiş ürünlere benzer ürünleri topla
    score = np.zeros_like(pop, dtype=np.float32)
    for i in hist_idx:
        score += sims[i]
    # Ziyaret edilenleri filtrele
    score[hist_idx] = -1e9
    top_idx = topk_from_scores(score, k)
    return [idx2item[int(i)] for i in top_idx]

def recommend_similar(product_id: int, k: int = 10) -> List[int]:
    products, pop, sims, tfidf, vect, item_maps, user_maps = load_artifacts()
    item2idx, idx2item = item_maps["item2idx"], item_maps["idx2item"]
    if product_id not in item2idx:
        raise KeyError("Unknown product_id")
    i = item2idx[product_id]
    scores = sims[i]
    scores[i] = -1e9
    top_idx = topk_from_scores(scores, k)
    return [idx2item[int(j)] for j in top_idx]

def recommend_content(product_id: int, k: int = 10) -> List[int]:
    products, pop, sims, tfidf, vect, item_maps, user_maps = load_artifacts()
    item2idx, idx2item = item_maps["item2idx"], item_maps["idx2item"]
    if product_id not in item2idx:
        raise KeyError("Unknown product_id")
    i = item2idx[product_id]
    sims_c = cosine_similarity(tfidf[i].reshape(1, -1), tfidf).ravel()
    sims_c[i] = -1e9
    top_idx = topk_from_scores(sims_c, k)
    return [idx2item[int(j)] for j in top_idx]

# ---------------------------
# 5) FastAPI Servisi
# ---------------------------

app = FastAPI(title="E-Commerce Recommender", version="1.0")

@app.get("/")
def root():
    return {
        "message": "E-Commerce Recommender API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "train": "POST /train",
            "user_recommendations": "/recommend/user/{user_id}?k=10",
            "product_recommendations": "/recommend/product/{product_id}?k=10&strategy=itemcf|content",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/train")
def train():
    stats = train_and_persist()
    return {"status": "trained", "stats": stats}

@app.get("/recommend/user/{user_id}", response_model=RecResponse)
def api_rec_user(user_id: int, k: int = Query(10, ge=1, le=100)):
    try:
        items = recommend_user(user_id, k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    prods = pd.read_csv(PRODUCTS_PATH).set_index("product_id")
    recs = []
    for pid in items:
        r = prods.loc[pid]
        recs.append({"product_id": int(pid), "title": str(r["title"]),
                     "brand": str(r["brand"]), "category": str(r["category"]),
                     "price": float(r["price"])})
    return RecResponse(user_id=user_id, recs=recs)

@app.get("/recommend/product/{product_id}", response_model=RecResponse)
def api_rec_similar(product_id: int, k: int = Query(10, ge=1, le=100), strategy: str = Query("itemcf", pattern="^(itemcf|content)$")):
    try:
        if strategy == "itemcf":
            items = recommend_similar(product_id, k)
        else:
            items = recommend_content(product_id, k)
    except KeyError as _:
        raise HTTPException(status_code=404, detail="Unknown product_id")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    prods = pd.read_csv(PRODUCTS_PATH).set_index("product_id")
    recs = []
    for pid in items:
        r = prods.loc[pid]
        recs.append({"product_id": int(pid), "title": str(r["title"]),
                     "brand": str(r["brand"]), "category": str(r["category"]),
                     "price": float(r["price"])})
    return RecResponse(product_id=product_id, recs=recs)

# Otomatik ilk eğitim (artifakt yoksa)
if __name__ == "__main__":
    if not (POPULARITY_PATH.exists() and ITEM_SIMS_PATH.exists() and TFIDF_MTX_PATH.exists() and ITEM_MAP_PATH.exists() and USER_MAP_PATH.exists()):
        print("Artifacts not found. Training once...")
        train_and_persist()
    print("Run API with: uvicorn main:app --reload")
