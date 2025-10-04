# ⚽ Futbol xG (Expected Goals) Modeli

Bu proje, futbol maçlarındaki şutların gol olma olasılığını tahmin eden gelişmiş bir makine öğrenmesi modeli sunar. Model, şut anındaki durumsal faktörlere dayalı **gol beklentisi (xG)** hesaplaması yapar.

## 🎯 Proje Amacı

- **Amaç**: Bir şutun gol olma ihtimalini öngören sınıflandırma modeli geliştirmek
- **Kapsam**: `events.csv` veri setinden şut olaylarını analiz etme
- **Hedef**: Oyuncu performansını hem gol oranı hem de şutlarının ortalama kalitesine (xG) göre değerlendirme

## 🚀 Hızlı Başlangıç

```bash
# Modeli çalıştırın
python training.py
```

## 📊 Model Özellikleri

### Kullanılan Algoritma
- **RandomForestClassifier** 
- `n_estimators=300`
- `min_samples_leaf=2`
- `class_weight="balanced"`
- `random_state=42`

### Özellikler (Features)

| Tip | Özellikler | Açıklama |
|-----|------------|----------|
| **Sayısal** | `time` | Maç zamanı |
| **Kategorik** | `shot_place`, `shot_outcome`, `location`, `bodypart`, `assist_method`, `situation`, `fast_break` | Şut detayları |

### Veri İşleme Pipeline

| Özellik Tipi | İşlem Adımları | Açıklama |
|--------------|----------------|----------|
| **Sayısal** | `SimpleImputer(strategy="median")` | Eksik değerler medyan ile doldurulur |
| **Kategorik** | `SimpleImputer` + `OneHotEncoder` | Eksik değerler en sık değerle doldurulur, One-Hot encoding uygulanır |

## 📈 Model Performansı

- **Metrik**: ROC-AUC Score
- **Beklenen Performans**: 0.80-0.85
- **Eğitim/Test Oranı**: 80% / 20%
- **Stratified Split**: Sınıf dengesini korur

## 📁 Çıktılar

### Model Dosyaları
- `goal_model.pkl` - Eğitilmiş RandomForest pipeline modeli

### Analiz Dosyaları  
- `player_stats.csv` - Oyuncu bazlı istatistikler:
  - `shots_count`: Şut sayısı
  - `goals`: Gol sayısı  
  - `goal_rate`: Gol oranı
  - `mean_model_proba`: Ortalama xG değeri

### Konsol Çıktıları
- ROC-AUC skoru
- En yüksek ortalama gol olasılığına sahip ilk 10 oyuncu

## 🔍 Oyuncu Performans Analizi

### Analiz Kriterleri
- En az **10 şut** atan oyuncular dahil edilir
- Oyuncular `mean_model_proba` (ortalama xG) değerine göre sıralanır
- Hem gol sayısı hem de pozisyon kalitesi değerlendirilir

### Oyuncu Bilgisi Çıkarımı
1. Veri setinde `player` veya `player_name` sütunu aranır
2. Bulunamazsa `text` sütunundan regex ile oyuncu adı çıkartılır
3. `player_extracted` sütunu oluşturulur

## 🛠️ Teknik Detaylar

### Veri Yükleme
```python
CANDIDATE_PATHS = [
    "events.csv",
    "data/events.csv", 
    "../data/events.csv",
    # Diğer olası yollar...
]
```

### Model Pipeline
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Pipeline otomatik olarak oluşturulur
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(...)),
    ('classifier', RandomForestClassifier(...))
])
```

### Performans Değerlendirme
```python
from sklearn.metrics import roc_auc_score

# Model değerlendirmesi
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc_score:.4f}")
```

## 📋 Gereksinimler

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
```

## 🎯 Kullanım Alanları

### Futbol Analitik
- **Oyuncu Değerlendirmesi**: Sadece gol sayısına değil, pozisyon kalitesine dayalı analiz
- **Transfer Analizi**: Oyuncuların gerçek potansiyelini ölçme
- **Takım Stratejisi**: Hangi pozisyonlardan daha etkili şutlar atıldığını anlama

### Performans Metrikleri
- **xG (Expected Goals)**: Bir oyuncunun beklenen gol sayısı
- **xG per Shot**: Şut başına ortalama gol beklentisi  
- **Overperformance/Underperformance**: Gerçek gol sayısı vs beklenen gol sayısı

## 🔧 Gelişmiş Özellikler

### Otomatik Veri Keşfi
- Eksik sütunlar otomatik olarak atlanır
- Farklı veri formatlarına uyum sağlar
- Hata durumlarında detaylı bilgi verir

### Sağlam Model Pipeline
- Eksik veri işleme
- Kategorik veri kodlama
- Sınıf dengesizliği düzeltme
- Cross-validation desteği

## 📊 Örnek Çıktı

```
ROC-AUC Score: 0.8234

Top 10 Players by Expected Goals per Shot:
1. Messi - xG/Shot: 0.234, Goals: 45, Shots: 156
2. Ronaldo - xG/Shot: 0.221, Goals: 38, Shots: 142
3. Lewandowski - xG/Shot: 0.218, Goals: 41, Shots: 167
...
```

## 🚀 Gelecek Geliştirmeler

### Model İyileştirmeleri
- [ ] XGBoost ve LightGBM desteği
- [ ] Deep Learning modelleri (Neural Networks)
- [ ] Ensemble methods (Voting, Stacking)
- [ ] Hyperparameter optimization (GridSearch, Bayesian)

### Özellik Geliştirmeleri  
- [ ] Şut açısı ve mesafe hesaplama
- [ ] Kaleci pozisyonu analizi
- [ ] Savunma baskısı faktörü
- [ ] Maç durumu (skor, zaman) etkisi

### Analiz Geliştirmeleri
- [ ] Takım bazlı xG analizi
- [ ] Maç bazlı momentum analizi
- [ ] Sezon boyunca performans trendi
- [ ] Rakip takım etkisi analizi

---

**Not**: Bu model eğitim ve araştırma amaçlıdır. Profesyonel futbol analitiği için ek validasyon ve domain expertise gereklidir.
