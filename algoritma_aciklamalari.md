# 🧠 E-TİCARET ÖNERİ SİSTEMİ - ALGORİTMA AÇIKLAMALARI

## 📊 Genel Sistem Mimarisi

```
[Kullanıcı Etkileşimleri] + [Ürün Özellikleri]
                    ↓
            [3 Farklı Algoritma]
                    ↓
              [Öneriler Sunumu]
```

---

## 🔥 ALGORİTMA 1: POPÜLERLİK BAZLI ÖNERİ

### 🎯 **Ne Yapar?**
En çok etkileşim alan ürünleri bulur ve önerir.

### 🧮 **Nasıl Hesaplar?**
```python
# Her ürün için toplam skor hesapla:
Ürün Skoru = (Görüntüleme × 1) + (Sepete Ekleme × 3) + (Satın Alma × 5)

# Örnek:
iPhone: 100 görüntüleme + 20 sepet + 10 satın alma
Skor = (100×1) + (20×3) + (10×5) = 100 + 60 + 50 = 210 puan
```

### 📈 **Kullanım Alanları:**
- ✅ Yeni kullanıcılar (hiç etkileşimi yok)
- ✅ Ana sayfa "Popüler Ürünler" bölümü
- ✅ Soğuk başlangıç problemi çözümü

### 💡 **Avantajları:**
- Çok hızlı hesaplama
- Basit ve anlaşılır
- Her zaman çalışır

### ⚠️ **Dezavantajları:**
- Kişiselleştirme yok
- Yeni ürünler dezavantajlı
- Çeşitlilik az

---

## 🤝 ALGORİTMA 2: ITEM COLLABORATIVE FILTERING

### 🎯 **Ne Yapar?**
"Bu ürünü alanlar şunları da aldı" mantığıyla çalışır.

### 🧮 **Nasıl Hesaplar?**

#### Adım 1: Kullanıcı-Ürün Matrisi Oluştur
```
        iPhone  AirPods  Kılıf  Şarj
Ahmet     5       0       3     0
Mehmet    4       5       4     2
Ayşe      0       4       0     5
Fatma     3       3       5     1
```

#### Adım 2: Ürünler Arası Benzerlik Hesapla
```python
# Kosinüs benzerliği kullan
iPhone ile AirPods benzerliği = cos(iPhone_vektörü, AirPods_vektörü)

# Sonuç matris:
           iPhone  AirPods  Kılıf  Şarj
iPhone      1.0     0.8     0.9   0.3
AirPods     0.8     1.0     0.7   0.9
Kılıf       0.9     0.7     1.0   0.2
Şarj        0.3     0.9     0.2   1.0
```

#### Adım 3: Öneri Üret
```python
# Kullanıcı iPhone aldıysa:
# iPhone'a en benzer ürünler: Kılıf (0.9), AirPods (0.8)
# Öneri: "iPhone Kılıfı ve AirPods'u da beğenebilirsin!"
```

### 📈 **Kullanım Alanları:**
- ✅ "Benzer ürünler" bölümü
- ✅ Sepet önerileri
- ✅ Kişisel ana sayfa

### 💡 **Avantajları:**
- Yüksek doğruluk
- Gerçek kullanıcı davranışı bazlı
- Sürpriz keşifler

### ⚠️ **Dezavantajları:**
- Yeni ürünler için zor
- Hesaplama yoğun
- Veri seyrekliği problemi

---

## 📝 ALGORİTMA 3: CONTENT-BASED FILTERING

### 🎯 **Ne Yapar?**
Ürün özelliklerine göre benzerlik bulur.

### 🧮 **Nasıl Hesaplar?**

#### Adım 1: Ürün Özelliklerini Metin Haline Getir
```python
iPhone 13: "Apple iPhone smartphone wireless 5G premium"
iPhone 14: "Apple iPhone smartphone wireless 5G premium latest"
Samsung S23: "Samsung Galaxy smartphone android wireless premium"
```

#### Adım 2: TF-IDF ile Sayısal Vektöre Çevir
```python
# TF-IDF: Kelimelerin önemini hesaplar
iPhone 13 vektörü: [Apple:0.8, iPhone:0.9, smartphone:0.6, wireless:0.4, ...]
iPhone 14 vektörü: [Apple:0.8, iPhone:0.9, smartphone:0.6, latest:0.7, ...]
```

#### Adım 3: Kosinüs Benzerliği Hesapla
```python
iPhone 13 vs iPhone 14 = 0.95 (çok benzer)
iPhone 13 vs Samsung S23 = 0.60 (orta benzer)
iPhone 13 vs Ayakkabı = 0.05 (hiç benzer değil)
```

### 📈 **Kullanım Alanları:**
- ✅ "Benzer özellikli ürünler"
- ✅ Kategori bazlı öneriler
- ✅ Marka sadakati

### 💡 **Avantajları:**
- Yeni ürünler için çalışır
- Açıklanabilir öneriler
- Veri seyrekliği problemi yok

### ⚠️ **Dezavantajları:**
- Özellik mühendisliği gerekli
- Sürpriz keşif az
- Aşırı uzmanlaşma riski

---

## 🎯 KARMA STRATEJİ: NASIL BİRLEŞTİRİLİR?

### Kullanıcı Durumuna Göre Algoritma Seçimi:

```python
def recommend_user(user_id):
    if user_is_new(user_id):
        return popularity_based_recommendations()  # Popüler ürünler
    elif user_has_few_interactions(user_id):
        return mix(popularity, content_based)      # Karma yaklaşım
    else:
        return collaborative_filtering()           # Kişisel öneriler
```

### Ürün Sayfasında Çeşitli Öneriler:

```python
def product_page_recommendations(product_id):
    return {
        "similar_behavior": item_collaborative_filtering(product_id),    # "Alanlar bunları da aldı"
        "similar_features": content_based_filtering(product_id),        # "Benzer özellikli"
        "trending": popularity_based_in_category(product_id.category)   # "Kategoride popüler"
    }
```

---

## 📊 PERFORMANS VE OPTİMİZASYON

### Hesaplama Karmaşıklığı:
- **Popülerlik:** O(n) - Çok hızlı
- **ItemCF:** O(n²) - Orta hızlı  
- **Content:** O(n×m) - Hızlı (m = özellik sayısı)

### Bellek Kullanımı:
- **Popülerlik:** Çok az (sadece skor dizisi)
- **ItemCF:** Orta (n×n benzerlik matrisi)
- **Content:** Az (TF-IDF matrisi)

### Gerçek Zamanlı Optimizasyonlar:
```python
# Önceden hesaplanmış modeller kullan
models = load_precomputed_models()

# Sadece top-k sonuçları hesapla
def fast_topk(scores, k=10):
    return np.argpartition(-scores, k-1)[:k]

# Cache mekanizması
@lru_cache(maxsize=1000)
def cached_recommendations(user_id, k):
    return compute_recommendations(user_id, k)
```

---

## 🎪 ÖRNEK SENARYOLAR

### Senaryo 1: Yeni Kullanıcı (Ahmet)
```
Ahmet ilk kez siteye girdi
→ Popülerlik algoritması devreye girer
→ En çok satılan ürünler önerilir
→ Sonuç: iPhone, AirPods, MacBook
```

### Senaryo 2: Aktif Kullanıcı (Ayşe)
```
Ayşe'nin geçmişi: Spor ayakkabısı, yoga matı, protein tozu
→ ItemCF algoritması devreye girer
→ Benzer kullanıcıların aldığı ürünler bulunur
→ Sonuç: Spor kıyafetleri, fitness tracker, vitamin
```

### Senaryo 3: Ürün Sayfası (iPhone 13)
```
Kullanıcı iPhone 13 sayfasında
→ Content-based: iPhone 14, iPhone 12 (benzer özellik)
→ ItemCF: AirPods, iPhone kılıfı (beraber alınan)
→ Popülerlik: En çok satan aksesuarlar
```

Bu sistem sayesinde her kullanıcıya en uygun önerileri sunabiliyoruz! 🚀
