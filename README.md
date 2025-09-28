---

Bu proje, futbol maçlarındaki şutların gol olma olasılığını tahmin eden bir **makine öğrenmesi modeli** ve bu tahminleri kullanarak oyuncu performansını analiz eden bir süreç sunar.

`training.py` betiği, ham olay verilerinden başlayarak, şut anındaki durumsal faktörlere dayalı bir **gol beklentisi (xG)** modeli kurar. Süreç; veri yükleme, filtreleme, özellik mühendisliği, model eğitimi ve oyuncu analizi adımlarını içerir.

Model olarak **RandomForestClassifier** kullanılmış, sınıf dengesizliği ise `class_weight="balanced"` parametresi ile giderilmiştir. Eksik veri doldurma ve kategorik verilerin kodlanması gibi ön işleme adımları, Pipeline yapısı ile otomatikleştirilmiştir. Performans ölçümünde **ROC-AUC** metriği tercih edilmiştir.

Sonuç olarak sadece bir model (`goal_model.pkl`) değil, aynı zamanda oyuncuların şutlarının ortalama gol olasılığına göre sıralandığı bir istatistik çıktısı (`player_stats.csv`) üretilmektedir. Bu sayede bir oyuncunun performansı, yalnızca attığı gol sayısıyla değil, girdiği pozisyonların kalitesiyle de değerlendirilmektedir.

---

## 1. Projenin Amacı ve Kapsamı

* **Amaç**: Bir şutun gol olma ihtimalini öngören sınıflandırma modeli geliştirmek.
* **Kapsam**: `events.csv` veri setinden yalnızca şutlar (event_type == 1) analiz edilmiştir.
* **Hedef**: Oyuncu performansını hem gol oranı hem de şutlarının ortalama kalitesine (xG) göre değerlendirmek.

---

## 2. Veri Hazırlık Süreci

### 2.1 Veri Yükleme

* Betik, farklı ortamlar için tanımlanan `CANDIDATE_PATHS` listesinden veri setini arar.
* Dosya bulunamazsa `FileNotFoundError` döner.

### 2.2 Veri Filtreleme

* Sadece şut olayları (`event_type == 1`) seçilir.
* `event_type` sütununun varlığı kontrol edilir.

### 2.3 Özellik ve Hedef Değişken

* **Hedef (y):** `is_goal` (1: Gol, 0: Gol değil).
* **Özellikler (X):**

  * Sayısal: `time` (maç zamanı).
  * Kategorik: `shot_place`, `shot_outcome`, `location`, `bodypart`, `assist_method`, `situation`, `fast_break`.
  * Veri setinde mevcut olmayan sütunlar otomatik atlanır.

---

## 3. Modelleme Mimarisi

### 3.1 Ön İşleme

| Özellik Tipi | Adımlar                                                                                     | Açıklama                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Sayısal      | `SimpleImputer(strategy="median")`                                                          | Eksikler medyan ile doldurulur.                                                                                            |
| Kategorik    | 1. `SimpleImputer(strategy="most_frequent")`<br>2. `OneHotEncoder(handle_unknown="ignore")` | Eksikler en sık görülen değerle doldurulur, ardından One-Hot Encoding yapılır. Yeni kategoriler hata vermeden yok sayılır. |

Bu dönüşümler **ColumnTransformer** ile birleştirilir.

### 3.2 Model Seçimi

* Algoritma: `RandomForestClassifier`
* Parametreler:

  * `n_estimators=300`
  * `min_samples_leaf=2`
  * `class_weight="balanced"`
  * `random_state=42`

### 3.3 Eğitim ve Değerlendirme

* Eğitim/Test bölünmesi: %80 / %20 (`stratify=y` ile).
* Eğitim: Pipeline fit edilir → hem ön işleme hem model çalışır.
* Değerlendirme: `roc_auc_score` metriği ile test setinde ölçülür.

---

## 4. Oyuncu Performans Analizi

### 4.1 Oyuncu Bilgisi

* Veri setinde `player`, `player_name` gibi sütun aranır.
* Bulunamazsa `text` sütunundan regex ile oyuncu adı çıkartılır (`player_extracted`).

### 4.2 İstatistikler

Her oyuncu için:

* `shots_count`: Şut sayısı
* `goals`: Gol sayısı
* `goal_rate`: Gol oranı
* `mean_model_proba`: Modelin verdiği ortalama gol olasılığı (xG)

### 4.3 Filtreleme

* En az **10 şut** atan oyuncular alınır.
* `mean_model_proba`’ya göre azalan sırada listelenir.

---

## 5. Çıktılar

* **goal_model.pkl** → Pipeline + eğitilmiş RandomForest modeli
* **player_stats.csv** → Oyuncu bazlı istatistikler
* Konsola ek olarak:

  * ROC-AUC skoru
  * En yüksek ortalama gol olasılığına sahip ilk 10 oyuncu

---

✅ Bu proje, sadece basit bir tahmin modeli değil, aynı zamanda **oyuncuların pozisyon kalitesine dayalı gelişmiş performans analizi** için bir temel sunmaktadır.

---