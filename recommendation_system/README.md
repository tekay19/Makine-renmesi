# 🔍 Öneri Sistemi - Netflix ve Amazon Nasıl Yapıyor Merakım

Merhaba! Bu proje, Netflix'in nasıl film önerdiğini, Amazon'un nasıl ürün önerdiğini merak etmemden doğdu. "Acaba ben de böyle bir sistem yapabilir miyim?" diye düşündüm ve işte karşınızda!

İki farklı yöntemi birleştirerek hibrit bir öneri sistemi yaptım. Hem kullanıcıların geçmiş davranışlarına bakıyor, hem de ürünlerin özelliklerini analiz ediyor. Böylece daha akıllı öneriler verebiliyor.

## 🎯 Neden Bu Sistemi Geliştirdim?

- **Merakım**: E-ticaret sitelerinin öneri algoritmaları nasıl çalışıyor?
- **Öğrenmek İstediklerim**: Collaborative filtering ve content-based filtering nedir?
- **Hedefim**: Kullanıcıları gerçekten ilgilendirecek öneriler verebilmek

## 🚀 Nasıl Çalıştırırsınız?

```bash
# Temel sistemi çalıştırmak için:
python main.py

# Eğer kodun nasıl çalıştığını anlamak istiyorsanız:
python main_commented.py  # Bu versiyonda her şeyi açıkladım!
```

## 📊 Sistemim Nasıl Çalışıyor?

### İki Farklı Yaklaşımı Birleştirdim

| Hangi Yöntem | Nasıl Çalışıyor | Neden Kullandım |
|--------------|-----------------|-----------------|
| **Collaborative Filtering** | "Sana benzer kullanıcılar ne aldı?" | Sosyal etkiyi yakalıyor, trendleri buluyor |
| **Content-Based Filtering** | "Bu ürüne benzer ürünler neler?" | Yeni ürünler için de çalışıyor |
| **Hibrit Sistem** | İkisini akıllıca birleştirdim | Her iki yöntemin güçlü yanlarını kullanıyor |

### Kullanılan Algoritmalar

1. **TF-IDF Vectorization**: Ürün açıklamalarından özellik çıkarımı
2. **Cosine Similarity**: Ürün benzerlik hesaplama
3. **Matrix Factorization**: Kullanıcı-ürün etkileşim analizi
4. **Popularity-Based**: Popüler ürün önerileri

## 📁 Dosya Yapısı

```
recommendation_system/
├── 📄 main.py                 # Ana öneri sistemi
├── 📄 main_commented.py       # Detaylı açıklamalı versiyon
├── 📄 README.md              # Bu dosya
├── 📁 data/                  # Veri dosyaları
│   ├── interactions.csv      # Kullanıcı-ürün etkileşimleri
│   └── products.csv          # Ürün bilgileri
└── 📁 artifacts/             # Model çıktıları
    ├── item_map.joblib       # Ürün ID haritası
    ├── user_map.joblib       # Kullanıcı ID haritası
    ├── item_sims.npy         # Ürün benzerlik matrisi
    ├── tfidf_vectorizer.joblib # TF-IDF vektörizer
    ├── tfidf.npy            # TF-IDF matrisi
    └── pop.npy              # Popülerlik skorları
```

## 🛠️ Teknik Özellikler

### Veri İşleme
```python
# Kullanıcı-ürün etkileşim matrisi
interactions_matrix = interactions.pivot_table(
    index='user_id', 
    columns='item_id', 
    values='rating',
    fill_value=0
)

# TF-IDF ile ürün özellik çıkarımı
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['description'])
```

### Benzerlik Hesaplama
```python
from sklearn.metrics.pairwise import cosine_similarity

# Ürün benzerlik matrisi
item_similarity = cosine_similarity(tfidf_matrix)

# Kullanıcı benzerlik matrisi  
user_similarity = cosine_similarity(interactions_matrix)
```

### Hibrit Öneri Algoritması
```python
def hybrid_recommend(user_id, n_recommendations=10):
    # Content-based öneriler
    content_recs = content_based_recommend(user_id)
    
    # Collaborative filtering öneriler
    collab_recs = collaborative_recommend(user_id)
    
    # Popülerlik tabanlı öneriler
    popular_recs = popularity_recommend()
    
    # Hibrit kombinasyon (ağırlıklı)
    final_recs = combine_recommendations(
        content_recs, collab_recs, popular_recs,
        weights=[0.4, 0.4, 0.2]
    )
    
    return final_recs[:n_recommendations]
```

## 📈 Performans Metrikleri

### Değerlendirme Kriterleri
- **Precision@K**: İlk K önerideki doğruluk oranı
- **Recall@K**: İlk K öneride yakalanan ilgili ürün oranı
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Diversity**: Öneri çeşitliliği
- **Coverage**: Katalog kapsama oranı

### Beklenen Performans
```
Precision@10: 0.15-0.25
Recall@10: 0.08-0.15
NDCG@10: 0.20-0.35
Diversity: 0.70-0.85
Coverage: 0.60-0.80
```

## 🔧 Konfigürasyon

### Sistem Parametreleri
```python
# TF-IDF parametreleri
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.8

# Öneri parametreleri
N_RECOMMENDATIONS = 10
MIN_INTERACTIONS = 5
SIMILARITY_THRESHOLD = 0.1

# Hibrit ağırlıklar
CONTENT_WEIGHT = 0.4
COLLABORATIVE_WEIGHT = 0.4
POPULARITY_WEIGHT = 0.2
```

### Veri Gereksinimleri
```python
# interactions.csv formatı
user_id,item_id,rating,timestamp
1,101,4.5,1609459200
1,102,3.0,1609545600

# products.csv formatı  
item_id,title,description,category,price
101,"Product A","Description of product A",Electronics,299.99
102,"Product B","Description of product B",Books,19.99
```

## 🎯 Kullanım Alanları

### E-ticaret
- **Ürün Önerileri**: Ana sayfa ve ürün detay sayfalarında
- **Cross-selling**: İlgili ürün önerileri
- **Up-selling**: Daha yüksek değerli alternatifler

### İçerik Platformları
- **Film/Müzik Önerileri**: Netflix, Spotify tarzı sistemler
- **Haber/Makale Önerileri**: Kişiselleştirilmiş içerik akışı
- **Sosyal Medya**: Friend/content suggestions

### İş Uygulamaları
- **B2B Ürün Önerileri**: Kurumsal alım süreçleri
- **İnsan Kaynakları**: İş-aday eşleştirme
- **Eğitim**: Kurs ve öğrenme materyali önerileri

## 🚀 Gelişmiş Özellikler

### Cold Start Problemi Çözümleri
1. **Yeni Kullanıcılar**: Popülerlik tabanlı öneriler
2. **Yeni Ürünler**: Content-based filtering
3. **Demografik Filtreleme**: Yaş, cinsiyet, lokasyon bazlı

### Gerçek Zamanlı Öneriler
```python
# Streaming güncellemeleri
def update_recommendations_realtime(user_id, item_id, rating):
    # Yeni etkileşimi sisteme ekle
    add_interaction(user_id, item_id, rating)
    
    # İlgili benzerlik matrislerini güncelle
    update_similarity_matrices(user_id, item_id)
    
    # Öneri cache'ini temizle
    clear_recommendation_cache(user_id)
```

### A/B Testing Desteği
```python
def ab_test_recommendations(user_id, test_group):
    if test_group == 'A':
        return content_based_recommend(user_id)
    elif test_group == 'B':
        return collaborative_recommend(user_id)
    else:
        return hybrid_recommend(user_id)
```

## 📊 Örnek Kullanım

### Temel Öneri Alma
```python
# Kullanıcı için öneri al
user_id = 123
recommendations = hybrid_recommend(user_id, n_recommendations=10)

print(f"User {user_id} için öneriler:")
for i, (item_id, score) in enumerate(recommendations, 1):
    print(f"{i}. Ürün {item_id} - Skor: {score:.3f}")
```

### Benzer Ürün Bulma
```python
# Belirli bir ürüne benzer ürünler
item_id = 101
similar_items = find_similar_items(item_id, n_items=5)

print(f"Ürün {item_id}'e benzer ürünler:")
for similar_id, similarity in similar_items:
    print(f"Ürün {similar_id} - Benzerlik: {similarity:.3f}")
```

### Performans Analizi
```python
# Sistem performansını değerlendir
metrics = evaluate_recommendations(test_interactions)
print(f"Precision@10: {metrics['precision_10']:.3f}")
print(f"Recall@10: {metrics['recall_10']:.3f}")
print(f"NDCG@10: {metrics['ndcg_10']:.3f}")
```

## 🔧 Kurulum ve Gereksinimler

### Python Paketleri
```bash
pip install pandas numpy scikit-learn joblib
```

### Detaylı Gereksinimler
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
scipy>=1.7.0
```

## 🚀 Gelecek Geliştirmeler

### Algoritma İyileştirmeleri
- [ ] **Deep Learning**: Neural Collaborative Filtering
- [ ] **Matrix Factorization**: SVD, NMF, ALS
- [ ] **Graph-based**: Node2Vec, GraphSAGE
- [ ] **Multi-armed Bandit**: Exploration vs Exploitation

### Sistem İyileştirmeleri
- [ ] **Distributed Computing**: Spark, Dask
- [ ] **Real-time Processing**: Kafka, Redis
- [ ] **Model Serving**: MLflow, TensorFlow Serving
- [ ] **Monitoring**: Prometheus, Grafana

### İş Değeri Artırımı
- [ ] **Explainable AI**: Öneri sebeplerini açıklama
- [ ] **Fairness**: Bias detection ve mitigation
- [ ] **Privacy**: Differential privacy, federated learning
- [ ] **Business Rules**: İş kuralları entegrasyonu

---

**Not**: Bu sistem demo amaçlıdır. Production kullanımı için ölçeklenebilirlik, güvenlik ve performans optimizasyonları gereklidir.
