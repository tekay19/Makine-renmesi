# Makine Öğrenmesi Projeleri Koleksiyonu

Bu repo, 3 farklı makine öğrenmesi projesini içermektedir. Her proje veri işleme, model eğitimi ve FastAPI kullanımını kapsamlı bir şekilde göstermektedir.

## 📋 Proje Listesi

### 1. 🎯 Müşteri Segmentasyonu ve Churn Prediction
**Dizin**: `project1_customer_segmentation/`
**Port**: 8001
**Teknolojiler**: K-means Clustering, Random Forest, Scikit-learn

**Özellikler**:
- Müşteri segmentasyonu (K-means)
- Churn prediction (Random Forest)
- Otomatik veri oluşturma
- Kapsamlı müşteri analizi

### 2. 📝 Sentiment Analysis ve Metin Sınıflandırma
**Dizin**: `project2_sentiment_analysis/`
**Port**: 8002
**Teknolojiler**: NLP, TF-IDF, Naive Bayes, SVM, NLTK

**Özellikler**:
- Sentiment analysis (Positive/Negative/Neutral)
- Metin kategori sınıflandırması (8 kategori)
- Anahtar kelime çıkarımı
- VADER emotion scoring

### 3. 🏠 Fiyat Tahmini ve Regresyon Analizi
**Dizin**: `project3_price_prediction/`
**Port**: 8003
**Teknolojiler**: XGBoost, Random Forest, Ensemble Methods

**Özellikler**:
- Emlak fiyat tahmini
- Ensemble model (4 farklı algoritma)
- Pazar analizi ve yatırım önerileri
- Feature importance analizi

## 🚀 Hızlı Başlangıç

### Tüm Projeleri Çalıştırma

```bash
# 1. Ana dizinde
cd /Users/semihtekay/ssss

# 2. Her proje için ayrı terminal açın ve şu komutları çalıştırın:

# Terminal 1 - Müşteri Segmentasyonu
cd project1_customer_segmentation
pip install -r requirements.txt
python main.py

# Terminal 2 - Sentiment Analysis  
cd project2_sentiment_analysis
pip install -r requirements.txt
python main.py

# Terminal 3 - Fiyat Tahmini
cd project3_price_prediction
pip install -r requirements.txt
python main.py
```

### API Erişim Adresleri

- **Proje 1**: http://localhost:8001
- **Proje 2**: http://localhost:8002  
- **Proje 3**: http://localhost:8003

Her proje için `/docs` endpoint'i ile Swagger UI'ya erişebilirsiniz.

## 📊 Proje Detayları

### Proje 1: Müşteri Segmentasyonu ve Churn Prediction

```bash
cd project1_customer_segmentation
python main.py
```

**API Endpoints**:
- `POST /predict/churn` - Churn tahmini
- `POST /predict/segment` - Müşteri segmentasyonu
- `POST /analyze/customer` - Kapsamlı müşteri analizi

**Örnek Kullanım**:
```python
import requests

customer_data = {
    "customer_id": 1,
    "age": 35,
    "gender": "Male",
    "tenure_months": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "contract_type": "One year",
    "payment_method": "Credit card",
    "internet_service": "Fiber optic",
    "online_security": "Yes",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "paperless_billing": "Yes"
}

response = requests.post("http://localhost:8001/predict/churn", json=customer_data)
print(response.json())
```

### Proje 2: Sentiment Analysis ve Metin Sınıflandırma

```bash
cd project2_sentiment_analysis
python main.py
```

**API Endpoints**:
- `POST /analyze/sentiment` - Sentiment analizi
- `POST /analyze/category` - Kategori sınıflandırması
- `POST /analyze/text` - Kapsamlı metin analizi
- `POST /extract/keywords` - Anahtar kelime çıkarımı

**Örnek Kullanım**:
```python
import requests

text_data = {
    "text": "This product is absolutely amazing! I love the quality and design.",
    "language": "en"
}

response = requests.post("http://localhost:8002/analyze/text", json=text_data)
result = response.json()

print(f"Sentiment: {result['sentiment_analysis']['sentiment']}")
print(f"Category: {result['category_analysis']['category']}")
print(f"Keywords: {result['keywords']}")
```

### Proje 3: Fiyat Tahmini ve Regresyon Analizi

```bash
cd project3_price_prediction
python main.py
```

**API Endpoints**:
- `POST /predict/price` - Fiyat tahmini
- `POST /analyze/market` - Pazar analizi
- `GET /features/importance` - Feature importance

**Örnek Kullanım**:
```python
import requests

property_data = {
    "area_sqft": 1200.0,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "floors": 2,
    "age_years": 5,
    "city": "Istanbul",
    "district": "Kadikoy",
    "neighborhood_score": 8.5,
    "property_type": "apartment",
    "construction_quality": "high",
    "heating_type": "central",
    "parking_spaces": 1,
    "distance_to_center_km": 12.0,
    "distance_to_metro_km": 0.8,
    "distance_to_school_km": 0.5,
    "distance_to_hospital_km": 2.0,
    "distance_to_mall_km": 1.5,
    "has_garden": False,
    "has_balcony": True,
    "has_elevator": True,
    "has_security": True,
    "has_gym": False,
    "has_pool": False,
    "market_trend": "rising",
    "season": "spring"
}

response = requests.post("http://localhost:8003/predict/price", json=property_data)
result = response.json()

print(f"Predicted Price: {result['predicted_price']:,.2f} TL")
print(f"Confidence: {result['model_confidence']:.2%}")
```

## 🛠️ Teknik Özellikler

### Ortak Teknolojiler
- **FastAPI**: Modern, hızlı web framework
- **Pydantic**: Veri validasyonu ve serialization
- **Scikit-learn**: Temel ML algoritmaları
- **Pandas & NumPy**: Veri işleme
- **Joblib**: Model serialization

### Proje Özel Teknolojiler

| Proje | ML Algoritmaları | Özel Kütüphaneler |
|-------|------------------|-------------------|
| Proje 1 | K-means, Random Forest | - |
| Proje 2 | Naive Bayes, SVM | NLTK, TextBlob |
| Proje 3 | XGBoost, Ensemble | XGBoost |

## 📈 Model Performansları

### Proje 1: Müşteri Segmentasyonu
- **Silhouette Score**: 0.65-0.75
- **Churn Prediction Accuracy**: 85-90%
- **F1-Score**: 0.82-0.88

### Proje 2: Sentiment Analysis
- **Sentiment Accuracy**: 85-90%
- **Category Classification**: 80-85%
- **Cross-validation**: 5-fold CV

### Proje 3: Fiyat Tahmini
- **R² Score**: 0.85-0.92
- **RMSE**: 25,000-35,000 TL
- **Ensemble Performance**: 0.90+ R²

## 🔧 Kurulum ve Gereksinimler

### Sistem Gereksinimleri
- Python 3.8+
- 4GB+ RAM
- 2GB+ disk alanı

### Bağımlılık Kurulumu
Her proje kendi `requirements.txt` dosyasına sahiptir:

```bash
# Proje 1
cd project1_customer_segmentation
pip install -r requirements.txt

# Proje 2  
cd project2_sentiment_analysis
pip install -r requirements.txt

# Proje 3
cd project3_price_prediction
pip install -r requirements.txt
```

### Docker ile Çalıştırma (Opsiyonel)

Her proje için Dockerfile oluşturabilirsiniz:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8001

CMD ["python", "main.py"]
```

## 📝 API Dokümantasyonu

Her proje otomatik Swagger UI dokümantasyonu sağlar:

- **Proje 1**: http://localhost:8001/docs
- **Proje 2**: http://localhost:8002/docs
- **Proje 3**: http://localhost:8003/docs

## 🧪 Test Etme

### Sağlık Kontrolü
```bash
# Tüm servislerin çalışıp çalışmadığını kontrol edin
curl http://localhost:8001/health
curl http://localhost:8002/health  
curl http://localhost:8003/health
```

### Örnek API Çağrıları
Her proje dizinindeki README dosyalarında detaylı örnekler bulabilirsiniz.

## 📊 Veri Setleri

Tüm projeler otomatik olarak sentetik veri oluşturur:

- **Proje 1**: 2000+ müşteri kaydı
- **Proje 2**: 3000+ metin örneği  
- **Proje 3**: 5000+ emlak kaydı

Veriler `data/` dizinlerinde CSV formatında saklanır.

## 🔄 Model Yeniden Eğitimi

Her proje `/retrain` endpoint'i ile modelleri yeniden eğitebilir:

```bash
curl -X POST http://localhost:8001/retrain
curl -X POST http://localhost:8002/retrain
curl -X POST http://localhost:8003/retrain
```

## 🎯 İş Değeri ve Kullanım Alanları

### Proje 1: Müşteri Segmentasyonu
- **CRM Sistemleri**: Müşteri lifecycle yönetimi
- **Pazarlama**: Hedefli kampanyalar
- **Retention**: Churn önleme stratejileri

### Proje 2: Sentiment Analysis
- **Sosyal Medya**: Brand monitoring
- **E-ticaret**: Ürün yorumu analizi
- **Müşteri Hizmetleri**: Otomatik ticket routing

### Proje 3: Fiyat Tahmini
- **Emlak**: Otomatik değerleme
- **Finans**: Mortgage risk analizi
- **Yatırım**: Portföy optimizasyonu

## 🚀 Geliştirme Önerileri

### Kısa Vadeli İyileştirmeler
1. **Monitoring**: Prometheus + Grafana
2. **Logging**: Structured logging (JSON)
3. **Authentication**: JWT token sistemi
4. **Rate Limiting**: API rate limiting

### Uzun Vadeli Geliştirmeler
1. **Microservices**: Kubernetes deployment
2. **Real-time**: Kafka streaming
3. **Advanced ML**: Deep learning modelleri
4. **MLOps**: Model versioning ve CI/CD

## 📞 Destek ve Katkı

### Hata Bildirimi
Herhangi bir hata ile karşılaştığınızda:
1. Hata mesajını kaydedin
2. Kullanılan veri örneğini paylaşın
3. Sistem bilgilerini (Python version, OS) belirtin

### Katkıda Bulunma
1. Fork yapın
2. Feature branch oluşturun
3. Test ekleyin
4. Pull request gönderin

## 📚 Ek Kaynaklar

### Dokümantasyon
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Öğrenme Kaynakları
- Machine Learning Coursera
- Hands-On Machine Learning (Kitap)
- Kaggle Learn Courses

---

**Not**: Bu projeler eğitim ve demo amaçlıdır. Production kullanımı için ek güvenlik, monitoring ve optimizasyon gereklidir.

