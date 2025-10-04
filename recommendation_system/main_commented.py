# =============================================================================
# E-TİCARET ÖNERİ SİSTEMİ - DETAYLI AÇIKLAMALI VERSİYON
# =============================================================================

# 📦 KÜTÜPHANE İMPORTLARI
# Bu kısım, programımızın ihtiyaç duyduğu araçları yükler
import os
import math
import random
from pathlib import Path  # Dosya yolları ile çalışmak için
from typing import List, Optional, Dict  # Tip belirtmek için (kod daha temiz olur)

import numpy as np  # Matematiksel işlemler için (matris hesaplamaları)
import pandas as pd  # Veri analizi için (Excel gibi tablolar)
from sklearn.feature_extraction.text import TfidfVectorizer  # Metin analizi için
from sklearn.metrics.pairwise import cosine_similarity  # Benzerlik hesaplama
from sklearn.preprocessing import normalize  # Veri normalizasyonu
from joblib import dump, load  # Model kaydetme/yükleme

from fastapi import FastAPI, HTTPException, Query  # Web API oluşturmak için
from pydantic import BaseModel  # Veri modelleri için

# 📁 DOSYA YOLLARI - Verilerimizin nerede saklanacağını belirtir
DATA_DIR = Path("data")  # Ham veri klasörü
INTERACTIONS_PATH = DATA_DIR / "interactions.csv"  # Kullanıcı etkileşimleri
PRODUCTS_PATH = DATA_DIR / "products.csv"  # Ürün bilgileri
ARTIFACT_DIR = Path("artifacts")  # Eğitilmiş modeller
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)  # Klasörü oluştur

# Model dosyalarının yolları
POPULARITY_PATH = ARTIFACT_DIR / "pop.npy"  # Popülerlik skorları
ITEM_SIMS_PATH = ARTIFACT_DIR / "item_sims.npy"  # Ürün benzerlik matrisi
USER_MAP_PATH = ARTIFACT_DIR / "user_map.joblib"  # Kullanıcı ID haritası
ITEM_MAP_PATH = ARTIFACT_DIR / "item_map.joblib"  # Ürün ID haritası
TFIDF_MTX_PATH = ARTIFACT_DIR / "tfidf.npy"  # Metin özellik matrisi
TFIDF_VECT_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"  # Metin vektörleyici

# =============================================================================
# 1️⃣ VERİ OLUŞTURMA VE YÜKLEME FONKSİYONLARI
# =============================================================================

def ensure_demo_data():
    """
    🎯 AMAÇ: Eğer veri yoksa, demo veri oluşturur
    
    Bu fonksiyon şunları yapar:
    1. Ürün verisi oluşturur (200 ürün)
    2. Kullanıcı etkileşim verisi oluşturur (32,000 etkileşim)
    """
    
    # Veri klasörünü oluştur
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    
    # ÜRÜN VERİSİ OLUŞTURMA
    if not PRODUCTS_PATH.exists():  # Eğer ürün dosyası yoksa
        print("📦 Ürün verisi oluşturuluyor...")
        
        rng = random.Random(42)  # Sabit seed (her seferinde aynı veri)
        brands = ["Acme", "Zen", "Nova", "Peak"]  # 4 marka
        cats = ["Shoe", "Tshirt", "PhoneCase", "Headphone", "Backpack"]  # 5 kategori
        
        rows = []  # Ürün listesi
        
        # 200 ürün oluştur (ID: 1-200)
        for pid in range(1, 201):
            brand = rng.choice(brands)  # Rastgele marka seç
            cat = rng.choice(cats)  # Rastgele kategori seç
            
            # Rastgele 3 etiket seç
            tags = " ".join(rng.sample([
                "sport", "leather", "casual", "running", "wireless", 
                "noise-cancel", "slim", "classic", "travel", "office", 
                "gaming", "summer", "winter"
            ], k=3))
            
            title = f"{brand} {cat} {pid}"  # Ürün başlığı
            price = round(rng.uniform(9.9, 299.9), 2)  # Rastgele fiyat
            
            # Ürünü listeye ekle
            rows.append([pid, title, brand, cat, tags, price])
        
        # DataFrame oluştur ve CSV'ye kaydet
        dfp = pd.DataFrame(rows, columns=["product_id", "title", "brand", "category", "tags", "price"])
        dfp.to_csv(PRODUCTS_PATH, index=False)
        print(f"✅ {len(rows)} ürün oluşturuldu!")

    # KULLANICI ETKİLEŞİM VERİSİ OLUŞTURMA
    if not INTERACTIONS_PATH.exists():  # Eğer etkileşim dosyası yoksa
        print("👥 Kullanıcı etkileşim verisi oluşturuluyor...")
        
        rng = random.Random(123)  # Farklı seed
        users = list(range(1, 401))  # 400 kullanıcı (ID: 1-400)
        prods = pd.read_csv(PRODUCTS_PATH)["product_id"].tolist()  # Ürün ID'leri
        
        # Etkileşim türleri ve ağırlıkları
        events = ["view", "cart", "purchase"]  # Görüntüleme, sepete ekleme, satın alma
        weights = {"view": 1.0, "cart": 3.0, "purchase": 5.0}  # Önem dereceleri
        
        rows = []  # Etkileşim listesi
        
        # Her kullanıcı için ~80 etkileşim oluştur
        for u in users:
            for _ in range(80):
                p = rng.choice(prods)  # Rastgele ürün seç
                # Etkileşim türünü seç (görüntüleme daha olası)
                e = random.choices(events, weights=[0.7, 0.2, 0.1], k=1)[0]
                ts = rng.randint(1_700_000_000, 1_750_000_000)  # Rastgele zaman
                
                rows.append([u, p, e, ts])
        
        # DataFrame oluştur ve CSV'ye kaydet
        dfi = pd.DataFrame(rows, columns=["user_id", "product_id", "event", "ts"])
        dfi.to_csv(INTERACTIONS_PATH, index=False)
        print(f"✅ {len(rows)} etkileşim oluşturuldu!")


def load_data():
    """
    🎯 AMAÇ: Veriyi yükler ve temizler
    
    RETURN: (products, interactions) - Temizlenmiş veri
    """
    ensure_demo_data()  # Önce veri var mı kontrol et
    
    # CSV dosyalarını oku
    products = pd.read_csv(PRODUCTS_PATH)
    interactions = pd.read_csv(INTERACTIONS_PATH)
    
    # VERİ TEMİZLİĞİ
    products = products.drop_duplicates("product_id").reset_index(drop=True)  # Duplikatları kaldır
    interactions = interactions.dropna(subset=["user_id", "product_id"])  # Boş değerleri kaldır
    
    # Veri tiplerini düzelt
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["product_id"] = interactions["product_id"].astype(int)
    
    return products, interactions


# =============================================================================
# 2️⃣ ÖNERİ ALGORİTMALARI - YARDIMCI FONKSİYONLAR
# =============================================================================

def build_mappings(items: pd.Series, users: pd.Series):
    """
    🎯 AMAÇ: ID'leri matris indekslerine çevirir
    
    Neden gerekli? 
    - Kullanıcı ID'leri: 5, 23, 156, 399... (düzensiz)
    - Matris indeksleri: 0, 1, 2, 3... (sıralı)
    
    RETURN: item2idx, idx2item, user2idx, idx2user
    """
    # Benzersiz ID'leri al ve sırala
    item_ids = np.sort(items.unique())  # [1, 2, 3, ..., 200]
    user_ids = np.sort(users.unique())  # [1, 2, 3, ..., 400]
    
    # ID -> İndeks haritaları
    item2idx = {pid: i for i, pid in enumerate(item_ids)}  # {1:0, 2:1, 3:2, ...}
    user2idx = {uid: i for i, uid in enumerate(user_ids)}  # {1:0, 2:1, 3:2, ...}
    
    # İndeks -> ID haritaları (ters çevirme)
    idx2item = {i: pid for pid, i in item2idx.items()}  # {0:1, 1:2, 2:3, ...}
    idx2user = {i: uid for uid, i in user2idx.items()}  # {0:1, 1:2, 2:3, ...}
    
    return item2idx, idx2item, user2idx, idx2user


def event_weight(e: str) -> float:
    """
    🎯 AMAÇ: Etkileşim türüne göre ağırlık verir
    
    Mantık:
    - Görüntüleme (view): 1.0 puan - Sadece baktı
    - Sepete ekleme (cart): 3.0 puan - İlgi gösterdi
    - Satın alma (purchase): 5.0 puan - Gerçekten beğendi
    """
    return {"view": 1.0, "cart": 3.0, "purchase": 5.0}.get(e, 0.5)


def user_item_matrix(interactions: pd.DataFrame, item2idx: Dict[int,int], user2idx: Dict[int,int]):
    """
    🎯 AMAÇ: Kullanıcı-Ürün etkileşim matrisi oluşturur
    
    Matris şekli:
    - Satırlar: Kullanıcılar (400 kullanıcı)
    - Sütunlar: Ürünler (200 ürün)
    - Değerler: Etkileşim skorları (0-5 arası)
    
    Örnek:
           Ürün1  Ürün2  Ürün3
    User1    0      3      5     <- User1, Ürün2'yi sepete ekledi, Ürün3'ü satın aldı
    User2    1      0      0     <- User2, Ürün1'i görüntüledi
    """
    n_users = len(user2idx)  # 400
    n_items = len(item2idx)  # 200
    
    # Sıfırlarla dolu matris oluştur
    UI = np.zeros((n_users, n_items), dtype=np.float32)
    
    # Her etkileşimi matrise ekle
    for _, row in interactions.iterrows():
        u = user2idx.get(row["user_id"])  # Kullanıcı indeksi
        i = item2idx.get(row["product_id"])  # Ürün indeksi
        
        if u is None or i is None:  # Geçersiz ID'leri atla
            continue
            
        # Etkileşim skorunu ekle
        UI[u, i] += event_weight(str(row["event"]))
    
    return UI


# =============================================================================
# 3️⃣ ALGORİTMA 1: POPÜLERLİK BAZLI ÖNERİ
# =============================================================================

def train_popularity(interactions: pd.DataFrame, item2idx: Dict[int,int]) -> np.ndarray:
    """
    🎯 AMAÇ: En popüler ürünleri bulur
    
    Mantık: Bir ürün ne kadar çok etkileşim alırsa, o kadar popülerdir
    
    Kullanım: Yeni kullanıcılar için (soğuk başlangıç problemi)
    """
    pop = np.zeros(len(item2idx), dtype=np.float32)  # Popülerlik skorları
    
    # Her etkileşimi say
    for _, row in interactions.iterrows():
        i = item2idx.get(row["product_id"])
        if i is None: 
            continue
        # Etkileşim ağırlığını ekle
        pop[i] += event_weight(str(row["event"]))
    
    return pop


# =============================================================================
# 4️⃣ ALGORİTMA 2: ITEM COLLABORATIVE FILTERING
# =============================================================================

def train_itemcf(UI: np.ndarray) -> np.ndarray:
    """
    🎯 AMAÇ: Ürünler arası benzerlik hesaplar
    
    Mantık: 
    - Aynı kullanıcılar tarafından beğenilen ürünler benzerdir
    - Kosinüs benzerliği kullanır (açı hesabı)
    
    Örnek:
    - Kullanıcı A: iPhone ve iPhone kılıfı aldı
    - Kullanıcı B: iPhone ve iPhone kılıfı aldı
    - Sonuç: iPhone ile iPhone kılıfı benzer
    """
    # Sütunları normalize et (her ürün için)
    I = normalize(UI, axis=0)  # Kullanıcı ekseninde L2 normalize
    
    # Ürün x Ürün benzerlik matrisi hesapla
    sims = I.T @ I  # Matrix çarpımı (200x200 matris)
    
    # Bir ürünün kendisiyle benzerliğini 0 yap
    np.fill_diagonal(sims, 0.0)
    
    return sims.astype(np.float32)


# =============================================================================
# 5️⃣ ALGORİTMA 3: CONTENT-BASED FILTERING
# =============================================================================

def train_content(products: pd.DataFrame):
    """
    🎯 AMAÇ: Ürün özelliklerine göre benzerlik hesaplar
    
    Mantık:
    - Aynı marka, kategori, etiketlere sahip ürünler benzerdir
    - TF-IDF ile metin analizi yapar
    
    Örnek:
    - "Acme Shoe sport leather" 
    - "Acme Shoe casual leather"
    - Sonuç: İki ürün benzer (aynı marka, kategori, ortak etiket)
    """
    # Tüm metin özelliklerini birleştir
    text = (products["title"].fillna("") + " " +
            products["brand"].fillna("") + " " +
            products["category"].fillna("") + " " +
            products["tags"].fillna("")).values
    
    # TF-IDF vektörleyici (metin -> sayısal vektör)
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vectorizer.fit_transform(text)  # Metin matrisini oluştur
    
    return X, vectorizer


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """
    🎯 AMAÇ: En yüksek k skoru olan indeksleri bulur
    
    Optimizasyon: argpartition kullanır (O(n) karmaşıklık)
    Normal sıralama O(n log n) olurdu
    """
    k = min(k, scores.shape[-1])  # k'yı sınırla
    if k <= 0:
        return np.array([], dtype=int)
    
    # En yüksek k değeri bul (kısmi sıralama)
    idx = np.argpartition(-scores, k-1)[:k]
    
    # Bulunanları tam sırala
    return idx[np.argsort(-scores[idx])]


# =============================================================================
# 6️⃣ MODEL EĞİTİMİ VE KAYDETME
# =============================================================================

def train_and_persist():
    """
    🎯 AMAÇ: Tüm modelleri eğitir ve diske kaydeder
    
    Adımlar:
    1. Veriyi yükle
    2. ID haritalarını oluştur
    3. Kullanıcı-Ürün matrisini oluştur
    4. 3 algoritmayı eğit
    5. Sonuçları kaydet
    """
    print("🚀 Model eğitimi başlıyor...")
    
    # 1. Veri yükleme
    products, interactions = load_data()
    print(f"📊 {len(products)} ürün, {len(interactions)} etkileşim yüklendi")
    
    # 2. ID haritaları
    item2idx, idx2item, user2idx, idx2user = build_mappings(
        products["product_id"], interactions["user_id"]
    )
    
    # 3. Kullanıcı-Ürün matrisi
    UI = user_item_matrix(interactions, item2idx, user2idx)
    print(f"📈 Kullanıcı-Ürün matrisi: {UI.shape}")

    # 4. ALGORİTMA EĞİTİMLERİ
    
    # Popülerlik modeli
    print("🔥 Popülerlik modeli eğitiliyor...")
    popularity = train_popularity(interactions, item2idx)
    np.save(POPULARITY_PATH, popularity)

    # Item-Item CF modeli
    print("🤝 Item Collaborative Filtering eğitiliyor...")
    item_sims = train_itemcf(UI)
    np.save(ITEM_SIMS_PATH, item_sims)

    # Content-based modeli
    print("📝 Content-based model eğitiliyor...")
    tfidf_mtx, vectorizer = train_content(products)
    np.save(TFIDF_MTX_PATH, tfidf_mtx.toarray().astype(np.float32))
    dump(vectorizer, TFIDF_VECT_PATH)

    # 5. Haritaları kaydet
    dump({"item2idx": item2idx, "idx2item": idx2item}, ITEM_MAP_PATH)
    dump({"user2idx": user2idx, "idx2user": idx2user}, USER_MAP_PATH)

    print("✅ Tüm modeller eğitildi ve kaydedildi!")
    
    return {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "ui_shape": list(UI.shape),
        "products": len(products),
        "interactions": len(interactions)
    }


# =============================================================================
# 7️⃣ ÖNERİ SUNMA FONKSİYONLARI
# =============================================================================

class RecResponse(BaseModel):
    """API yanıt modeli"""
    user_id: Optional[int] = None
    product_id: Optional[int] = None
    recs: List[Dict[str, object]]


def load_artifacts():
    """Kaydedilmiş modelleri yükler"""
    products = pd.read_csv(PRODUCTS_PATH)
    pop = np.load(POPULARITY_PATH, allow_pickle=False)
    sims = np.load(ITEM_SIMS_PATH, allow_pickle=False)
    tfidf = np.load(TFIDF_MTX_PATH, allow_pickle=False)
    vect = load(TFIDF_VECT_PATH)
    item_maps = load(ITEM_MAP_PATH)
    user_maps = load(USER_MAP_PATH)
    return products, pop, sims, tfidf, vect, item_maps, user_maps


def recommend_user(user_id: int, k: int = 10) -> List[int]:
    """
    🎯 AMAÇ: Kullanıcıya özel öneriler sunar
    
    Strateji:
    1. Kullanıcı sistemde var mı kontrol et
    2. Yoksa -> Popüler ürünleri öner (soğuk başlangıç)
    3. Varsa -> Geçmişine benzer ürünleri öner (ItemCF)
    """
    # Modelleri yükle
    products, pop, sims, tfidf, vect, item_maps, user_maps = load_artifacts()
    item2idx, idx2item = item_maps["item2idx"], item_maps["idx2item"]
    user2idx = user_maps["user2idx"]

    # DURUM 1: Yeni kullanıcı (soğuk başlangıç)
    if user_id not in user2idx:
        print(f"🆕 Yeni kullanıcı {user_id} - Popüler ürünler öneriliyor")
        top_idx = topk_from_scores(pop, k)
        return [idx2item[int(i)] for i in top_idx]

    # DURUM 2: Mevcut kullanıcı
    # Kullanıcının geçmişini al
    inter = pd.read_csv(INTERACTIONS_PATH)
    hist = inter.loc[inter["user_id"] == user_id, "product_id"].tolist()
    hist_idx = [item2idx[p] for p in hist if p in item2idx]
    
    if len(hist_idx) == 0:  # Etkileşimi yok
        top_idx = topk_from_scores(pop, k)
        return [idx2item[int(i)] for i in top_idx]

    # ItemCF: Geçmiş ürünlere benzer ürünleri topla
    print(f"🎯 Kullanıcı {user_id} için kişisel öneriler (geçmiş: {len(hist)} ürün)")
    score = np.zeros_like(pop, dtype=np.float32)
    
    # Her geçmiş ürün için benzer ürünleri topla
    for i in hist_idx:
        score += sims[i]  # Benzerlik skorlarını ekle
    
    # Zaten etkileşimde bulunduğu ürünleri filtrele
    score[hist_idx] = -1e9  # Çok düşük skor ver
    
    top_idx = topk_from_scores(score, k)
    return [idx2item[int(i)] for i in top_idx]


def recommend_similar(product_id: int, k: int = 10) -> List[int]:
    """
    🎯 AMAÇ: Bir ürüne benzer ürünleri bulur (ItemCF ile)
    
    Kullanım: "Bu ürünü beğenenler bunları da beğendi"
    """
    products, pop, sims, tfidf, vect, item_maps, user_maps = load_artifacts()
    item2idx, idx2item = item_maps["item2idx"], item_maps["idx2item"]
    
    if product_id not in item2idx:
        raise KeyError("Bilinmeyen ürün ID'si")
    
    i = item2idx[product_id]  # Ürün indeksi
    scores = sims[i]  # Bu ürünün tüm ürünlerle benzerliği
    scores[i] = -1e9  # Kendisini hariç tut
    
    top_idx = topk_from_scores(scores, k)
    return [idx2item[int(j)] for j in top_idx]


def recommend_content(product_id: int, k: int = 10) -> List[int]:
    """
    🎯 AMAÇ: Bir ürüne benzer ürünleri bulur (Content-based ile)
    
    Kullanım: "Benzer özellikli ürünler"
    """
    products, pop, sims, tfidf, vect, item_maps, user_maps = load_artifacts()
    item2idx, idx2item = item_maps["item2idx"], item_maps["idx2item"]
    
    if product_id not in item2idx:
        raise KeyError("Bilinmeyen ürün ID'si")
    
    i = item2idx[product_id]  # Ürün indeksi
    
    # Bu ürünün tüm ürünlerle içerik benzerliği
    sims_c = cosine_similarity(tfidf[i].reshape(1, -1), tfidf).ravel()
    sims_c[i] = -1e9  # Kendisini hariç tut
    
    top_idx = topk_from_scores(sims_c, k)
    return [idx2item[int(j)] for j in top_idx]


# =============================================================================
# 8️⃣ FASTAPI WEB SERVİSİ
# =============================================================================

app = FastAPI(title="E-Commerce Recommender", version="1.0")

@app.get("/")
def root():
    """Ana sayfa - API bilgileri"""
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
    """Sistem sağlık kontrolü"""
    return {"ok": True}

@app.post("/train")
def train():
    """Modelleri yeniden eğit"""
    stats = train_and_persist()
    return {"status": "trained", "stats": stats}

@app.get("/recommend/user/{user_id}", response_model=RecResponse)
def api_rec_user(user_id: int, k: int = Query(10, ge=1, le=100)):
    """
    🎯 Kullanıcıya özel öneriler
    
    Parametreler:
    - user_id: Kullanıcı ID'si
    - k: Öneri sayısı (1-100 arası)
    """
    try:
        items = recommend_user(user_id, k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Ürün detaylarını ekle
    prods = pd.read_csv(PRODUCTS_PATH).set_index("product_id")
    recs = []
    for pid in items:
        r = prods.loc[pid]
        recs.append({
            "product_id": int(pid), 
            "title": str(r["title"]),
            "brand": str(r["brand"]), 
            "category": str(r["category"]),
            "price": float(r["price"])
        })
    
    return RecResponse(user_id=user_id, recs=recs)

@app.get("/recommend/product/{product_id}", response_model=RecResponse)
def api_rec_similar(
    product_id: int, 
    k: int = Query(10, ge=1, le=100), 
    strategy: str = Query("itemcf", pattern="^(itemcf|content)$")
):
    """
    🎯 Ürüne benzer ürünler
    
    Parametreler:
    - product_id: Ürün ID'si
    - k: Öneri sayısı (1-100 arası)
    - strategy: itemcf (davranış bazlı) veya content (özellik bazlı)
    """
    try:
        if strategy == "itemcf":
            items = recommend_similar(product_id, k)
        else:
            items = recommend_content(product_id, k)
    except KeyError as _:
        raise HTTPException(status_code=404, detail="Bilinmeyen ürün ID'si")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Ürün detaylarını ekle
    prods = pd.read_csv(PRODUCTS_PATH).set_index("product_id")
    recs = []
    for pid in items:
        r = prods.loc[pid]
        recs.append({
            "product_id": int(pid), 
            "title": str(r["title"]),
            "brand": str(r["brand"]), 
            "category": str(r["category"]),
            "price": float(r["price"])
        })
    
    return RecResponse(product_id=product_id, recs=recs)

# =============================================================================
# 9️⃣ PROGRAM BAŞLATMA
# =============================================================================

if __name__ == "__main__":
    """
    Program ilk çalıştırıldığında:
    1. Model dosyaları var mı kontrol et
    2. Yoksa eğitimi başlat
    3. Kullanıcıya talimat ver
    """
    # Tüm model dosyaları mevcut mu?
    required_files = [
        POPULARITY_PATH, ITEM_SIMS_PATH, TFIDF_MTX_PATH, 
        ITEM_MAP_PATH, USER_MAP_PATH
    ]
    
    if not all(f.exists() for f in required_files):
        print("📚 Model dosyaları bulunamadı. İlk eğitim başlatılıyor...")
        train_and_persist()
    else:
        print("✅ Model dosyaları mevcut!")
    
    print("\n🚀 API'yi başlatmak için:")
    print("uvicorn main:app --reload")
    print("\n📖 Dokümantasyon: http://localhost:8000/docs")
