
# training.py
# Bu script, bir futbol şutunun gol olma olasılığını tahmin etmek için bir model eğitir
# ve modelin tahminlerine göre oyuncu performansını analiz eder.

# Gerekli kütüphaneleri içeri aktarma
# Veri manipülasyonu, modelleme vb. için gerekli kütüphaneleri içeri aktarma.
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier algoritması kullanılıyor
from sklearn.metrics import roc_auc_score
import joblib
import re

# --- 1) Veriyi yükle ---
# CSV dosyasından veriyi yükleme.
# Olası dosya yollarını tanımlama (Kaggle Notebook, yerel unzip veya aynı klasör)
CANDIDATE_PATHS = [
    "/kaggle/input/football-events/events.csv",     # Kaggle yarışma ortamı
    "football_events/events.csv",                   # yerel olarak zip'ten çıkarılmışsa
    "events.csv",                               # kodun çalıştığı klasörde
    "/root/.cache/kagglehub/datasets/secareanualin/football-events/versions/1/events.csv" # Kagglehub path
]
# Dosyanın bulunduğu ilk yolu bulma
data_path = None
for p in CANDIDATE_PATHS:
    if Path(p).exists():
        data_path = p
        break
else:
    # Dosya bulunamazsa hata verme
    raise FileNotFoundError("events.csv bulunamadı, doğru yolu verin.")

# CSV dosyasını pandas DataFrame olarak yükleme
df = pd.read_csv(data_path)

# --- 2) Sadece şutları al (event_type == 1) ---
# DataFrame'i sadece şutları içerecek şekilde filtreleme (event_type == 1).
# DataFrame'in 'event_type' sütununu içerip içermediğini kontrol etme
if "event_type" not in df.columns:
    raise KeyError("Bu dataset event_type sütunu içermiyor. Lütfen doğru CSV'yi kullandığından emin ol.")
# event_type'ı 1 olan satırları filtreleme (şutlar) ve kopyasını oluşturma
shots = df[df["event_type"] == 1].copy()

# --- 3) Zorunlu label ve temel kolonlar ---
# Model için hedef değişkeni (is_goal) ve özellikleri tanımlama.
# Hedef değişkenin ('is_goal') varlığını kontrol etme
if "is_goal" not in shots.columns:
    raise KeyError("is_goal kolonu yok. Bu sürümde etiket farklı isimde olabilir.")

# Tahmin için kullanılacak sayısal ve kategorik özellik adaylarını belirleme
# Bu sütunlar tipik olarak datasetin yapısına göre seçilir
features_num = []
if "time" in shots.columns:  # 'time' sütunu varsa sayısal özelliklere ekle
    features_num.append("time")

# Olası kategorik özelliklerden dataset'te bulunanları seçme
features_cat = [c for c in ["shot_place","shot_outcome","location","bodypart","assist_method","situation","fast_break"] if c in shots.columns]

# Model için en az bir özelliğin kullanılabilir olup olmadığını kontrol etme
assert (len(features_num) + len(features_cat)) > 0, "Kullanılabilir feature bulunamadı."

# Özellikleri (X) ve hedef değişkeni (y) belirleme
X = shots[features_num + features_cat].copy()
y = shots["is_goal"].astype(int).copy() # Hedef değişkeni integer tipine dönüştürme

# --- 4) NaN/Tip düzeni pipeline içinde halledilecek ---
# Sayısal ve kategorik özellikler için ön işleme adımlarını içeren pipeline'lar oluşturma.
# Sayısal özellikler için eksik değerleri medyan ile doldurma pipeline'ı
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])
# Kategorik özellikler için eksik değerleri en sık görülen değer ile doldurma ve One-Hot Encoding pipeline'ı
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")) # Bilinmeyen kategorileri göz ardı etme
])

# Farklı sütun tiplerine farklı dönüşümler uygulayan ColumnTransformer
# Farklı sütun tipleri için dönüştürücüleri birleştirme.
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, features_num), # Sayısal özelliklere sayısal transformer uygula
        ("cat", categorical_transformer, features_cat), # Kategorik özelliklere kategorik transformer uygula
    ],
    remainder="drop" # Belirtilmeyen sütunları bırakma
)

# RandomForestClassifier modelini tanımlama
# RandomForestClassifier modelini tanımlama. Bu, birden çok karar ağacı kullanan bir topluluk (ensemble) yöntemidir.
model = RandomForestClassifier(
    n_estimators=300, # Ormandaki ağaç sayısı
    min_samples_leaf=2, # Bir yaprak düğümde bulunması gereken minimum örnek sayısı
    class_weight="balanced", # Sınıf dengesizliğini ele almak için ağırlıklandırma
    random_state=42 # Tekrarlanabilirlik için rastgele durum
)

# Ön işleme ve modelleme adımlarını birleştiren ana pipeline
# Ön işleme ve modeli birleştiren bir pipeline oluşturma.
pipe = Pipeline(steps=[
    ("prep", preprocess), # Ön işleme adımı
    ("clf", model) # Sınıflandırıcı (model) adımı
])

# --- 5) Train / Test ---
# Veriyi eğitim ve test setlerine ayırma ve pipeline'ı eğitme.
# Veriyi eğitim (%80) ve test (%20) setlerine ayırma
# stratify=y: Hedef değişkenin dağılımını eğitim ve test setlerinde koruma
# random_state=42: Tekrarlanabilirlik için
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Pipeline'ı eğitim verisi üzerinde fit etme (hem ön işleme hem model eğitimi)
pipe.fit(X_train, y_train)
# Test seti üzerinde modelin tahmin olasılıklarını alma (pozitif sınıfın olasılığı)
proba = pipe.predict_proba(X_test)[:,1]
# ROC-AUC skorunu hesaplama (modelin performans metriği)
auc = roc_auc_score(y_test, proba)
# ROC-AUC skorunu yazdırma
print(f"ROC-AUC: {auc:.6f}")

# --- 6) Oyuncu bilgisini çıkarma ---
# Oyuncu bilgilerini çıkarma ve model tahminlerine göre performansı analiz etme.
# Oyuncu adını içeren sütunu bulmaya çalışma (farklı isimlerde olabilir)
player_col = None
for cand in ["player","player_name","playerId","player_id"]:
    if cand in shots.columns:
        player_col = cand
        break

# Eğer standart bir oyuncu sütunu bulunamazsa 'text' sütunundan oyuncu adını çıkarmaya çalışma
if player_col is None:
    # 'text' sütunundan oyuncu adını çıkarmak için yardımcı fonksiyon
    def extract_player_from_text(s):
        if pd.isna(s):
            return np.nan
        # Basit regex deseni ile oyuncu adını (Parantez içindeki takım adından önceki kısım) yakalama
        m = re.search(r"\.\s*([A-ZÇĞİÖŞÜ][^\(]+)\s*\(", s)  # Türkçe karakterleri de içerecek şekilde
        if m:
            name = m.group(1).strip()
            # Çok uzun veya anlamsız yakalamaları eleme
            if len(name.split()) <= 5:
                return name
        return np.nan
    # 'text' sütununa extract_player_from_text fonksiyonunu uygulama
    shots["player_extracted"] = shots["text"].apply(extract_player_from_text)
    player_col = "player_extracted" # Yeni sütunu oyuncu sütunu olarak belirleme

# Tüm şutlar için modelin tahmin olasılığını hesaplama ve yeni sütun olarak ekleme
# Not: pipeline fit edilmiş durumda, tüm X (train+test) üzerinden tahmin alıyoruz
all_proba = pipe.predict_proba(X)[:,1]
shots["_model_proba"] = all_proba

# Oyuncu bazında istatistik tablosu oluşturma
# Oyuncuya göre gruplama ve ortalama model olasılığı dahil istatistikleri toplama.
# Gruplama ve aggregate fonksiyonları ile şut sayısı, gol sayısı, gol oranı ve ortalama model olasılığını hesaplama
player_stats = shots.groupby(player_col).agg(
    shots_count=("is_goal","count"), # Her oyuncunun toplam şut sayısı
    goals=("is_goal","sum"), # Her oyuncunun toplam gol sayısı
    goal_rate=("is_goal","mean"), # Her oyuncunun gol oranı (goller / şutlar)
    mean_model_proba=("_model_proba","mean") # Her oyuncunun şutlarının ortalama model olasılığı
).reset_index().rename(columns={player_col:"player"}) # Index'i sıfırlama ve oyuncu sütununu yeniden adlandırma

# Analizi daha anlamlı kılmak için en az 10 şutu olan oyuncuları filtreleme
# Minimum sayıda şutu olan oyuncuları filtreleme ve ortalama model olasılığına göre sıralama.
player_stats = player_stats[player_stats["shots_count"] >= 10].sort_values("mean_model_proba", ascending=False) # Ortalama model olasılığına göre sıralama

# --- 7) Artefaktları kaydet ---
# Eğitilmiş modeli ve oyuncu istatistiklerini dosyalara kaydetme.
# Eğitilmiş model pipeline'ını dosya olarak kaydetme
joblib.dump(pipe, "goal_model.pkl")
# Oyuncu istatistikleri DataFrame'ini CSV dosyası olarak kaydetme
player_stats.to_csv("player_stats.csv", index=False)
# Kaydedilen dosyaları bildiren mesaj yazdırma
print("Kaydedildi: goal_model.pkl, player_stats.csv")

# En yüksek ortalama model olasılığına sahip ilk 10 oyuncuyu yazdırma
print(player_stats.head(10))