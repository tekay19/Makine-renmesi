# Proje 3: Fiyat Tahmini ve Regresyon Analizi

Bu proje, emlak verilerini kullanarak gelişmiş makine öğrenmesi teknikleri ile fiyat tahmini yapar ve kapsamlı pazar analizi sağlar.

## Özellikler

### 🏠 Kapsamlı Feature Engineering
- Temel emlak özellikleri (alan, oda sayısı, yaş)
- Lokasyon analizi (şehir, mahalle, mesafeler)
- Yapısal özellikler (kalite, tip, ısıtma)
- Çevresel faktörler (metro, okul, hastane mesafeleri)
- Lüks özellikler (havuz, güvenlik, spor salonu)
- Pazar dinamikleri (trend, mevsim)

### 🤖 Gelişmiş ML Modelleri
- **XGBoost**: Gradient boosting algoritması
- **Random Forest**: Ensemble tree-based model
- **Gradient Boosting**: Sequential boosting
- **Ridge Regression**: Regularized linear model
- **Ensemble Model**: Weighted model combination

### 📊 Pazar Analizi
- Fiyat tahmin aralığı ve güven skoru
- Karşılaştırmalı emlak analizi
- Yatırım önerileri
- Feature importance analizi

### 🚀 FastAPI Servisi
- RESTful API endpoints
- Gerçek zamanlı fiyat tahmini
- Kapsamlı pazar analizi
- Model performans metrikleri

## Kurulum

```bash
cd project3_price_prediction
pip install -r requirements.txt
```

## Kullanım

### API Servisini Başlatma
```bash
python main.py
```

API şu adreste çalışacak: http://localhost:8003

### API Endpoints

#### 1. Sağlık Kontrolü
```bash
GET /health
```

#### 2. Fiyat Tahmini
```bash
POST /predict/price
```

Örnek request body:
```json
{
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
  "has_garden": false,
  "has_balcony": true,
  "has_elevator": true,
  "has_security": true,
  "has_gym": false,
  "has_pool": false,
  "market_trend": "rising",
  "season": "spring"
}
```

Örnek response:
```json
{
  "predicted_price": 485000.75,
  "confidence_interval": [450000.25, 519000.25],
  "model_confidence": 0.8542,
  "price_per_sqft": 404.17,
  "market_analysis": {
    "value_assessment": "Fair",
    "investment_potential": "Good",
    "location_rating": "Good"
  },
  "feature_importance": {
    "area_sqft": 0.2845,
    "city": 0.1923,
    "neighborhood_score": 0.1456,
    "construction_quality": 0.1234,
    "distance_to_center_km": 0.0987
  }
}
```

#### 3. Pazar Analizi
```bash
POST /analyze/market
```

#### 4. Feature Importance
```bash
GET /features/importance?top_k=15
```

#### 5. Model Yeniden Eğitimi
```bash
POST /retrain
```

## Proje Yapısı

```
project3_price_prediction/
├── main.py                      # Ana uygulama dosyası
├── requirements.txt             # Python bağımlılıkları
├── README.md                   # Proje dokümantasyonu
├── data/                       # Veri dosyaları
│   └── property_data.csv      # Emlak verileri
├── models/                     # Eğitilmiş modeller
│   └── price_prediction_models.pkl # Kaydedilmiş ML modelleri
└── plots/                      # Görselleştirmeler
    └── (model performance plots)
```

## Teknik Detaylar

### Feature Engineering Pipeline
1. **Temel Özellikler**:
   - `price_per_sqft`: Metrekare başına fiyat
   - `rooms_per_floor`: Kat başına oda sayısı
   - `bathroom_bedroom_ratio`: Banyo/yatak odası oranı

2. **Kategorik Özellikler**:
   - `size_category`: Alan kategorisi (küçük, orta, büyük, çok büyük)
   - `age_category`: Yaş kategorisi (yeni, yakın, olgun, eski)

3. **Kompozit Skorlar**:
   - `location_score`: Lokasyon kalite skoru
   - `luxury_score`: Lüks özellik skoru
   - `total_rooms`: Toplam oda sayısı

### Makine Öğrenmesi Modelleri

#### XGBoost Regressor
- **Parametreler**: 100 estimator, max_depth=6, learning_rate=0.1
- **Avantajlar**: Yüksek performans, feature importance
- **Kullanım**: Ana tahmin modeli

#### Random Forest Regressor
- **Parametreler**: 100 estimator, max_depth=10
- **Avantajlar**: Overfitting direnci, robust predictions
- **Kullanım**: Ensemble component

#### Gradient Boosting Regressor
- **Parametreler**: 100 estimator, max_depth=6
- **Avantajlar**: Sequential learning, bias reduction
- **Kullanım**: Ensemble component

#### Ridge Regression
- **Parametreler**: alpha=1.0, standardized features
- **Avantajlar**: Linear relationships, interpretability
- **Kullanım**: Baseline ve ensemble component

### Ensemble Methodology
- **Weighted Average**: R² skorlarına göre ağırlıklandırma
- **Confidence Calculation**: Prediction variance based
- **Performance Metrics**: RMSE, MAE, R² score

### Veri Seti Özellikleri
- **Boyut**: 5000+ sentetik emlak kaydı
- **Şehirler**: İstanbul, Ankara, İzmir, Bursa, Antalya
- **Fiyat Aralığı**: 50,000 - 2,000,000 TL
- **Özellik Sayısı**: 25+ feature

## API Kullanım Örnekleri

### Python ile API Kullanımı
```python
import requests

# Fiyat tahmini
property_data = {
    "area_sqft": 1500.0,
    "bedrooms": 4,
    "bathrooms": 3.0,
    "floors": 2,
    "age_years": 8,
    "city": "Istanbul",
    "district": "Besiktas",
    "neighborhood_score": 9.0,
    "property_type": "apartment",
    "construction_quality": "luxury",
    "heating_type": "central",
    "parking_spaces": 2,
    "distance_to_center_km": 5.0,
    "distance_to_metro_km": 0.3,
    "distance_to_school_km": 0.2,
    "distance_to_hospital_km": 1.0,
    "distance_to_mall_km": 0.8,
    "has_garden": False,
    "has_balcony": True,
    "has_elevator": True,
    "has_security": True,
    "has_gym": True,
    "has_pool": True,
    "market_trend": "rising",
    "season": "summer"
}

response = requests.post("http://localhost:8003/predict/price", json=property_data)
result = response.json()

print(f"Predicted Price: {result['predicted_price']:,.2f} TL")
print(f"Price per sqft: {result['price_per_sqft']:.2f} TL")
print(f"Confidence: {result['model_confidence']:.2%}")

# Pazar analizi
market_response = requests.post("http://localhost:8003/analyze/market", json=property_data)
market_result = market_response.json()

print(f"Market Assessment: {market_result['market_value_assessment']}")
print(f"Investment Recommendation: {market_result['investment_recommendation']}")
```

### cURL ile API Kullanımı
```bash
# Fiyat tahmini
curl -X POST "http://localhost:8003/predict/price" \
     -H "Content-Type: application/json" \
     -d '{
       "area_sqft": 1200.0,
       "bedrooms": 3,
       "bathrooms": 2.0,
       "floors": 1,
       "age_years": 15,
       "city": "Ankara",
       "district": "Cankaya",
       "neighborhood_score": 7.5,
       "property_type": "house",
       "construction_quality": "medium",
       "heating_type": "individual",
       "parking_spaces": 1,
       "distance_to_center_km": 8.0,
       "distance_to_metro_km": 1.5,
       "distance_to_school_km": 0.8,
       "distance_to_hospital_km": 2.5,
       "distance_to_mall_km": 3.0,
       "has_garden": true,
       "has_balcony": true,
       "has_elevator": false,
       "has_security": false,
       "has_gym": false,
       "has_pool": false,
       "market_trend": "stable",
       "season": "autumn"
     }'

# Feature importance
curl -X GET "http://localhost:8003/features/importance?top_k=10"
```

## Model Performansı

### Performans Metrikleri
- **R² Score**: 0.85-0.92 (model türüne göre)
- **RMSE**: 25,000-35,000 TL
- **MAE**: 18,000-28,000 TL
- **Ensemble R²**: 0.90+ (tipik)

### Feature Importance (Tipik Sıralama)
1. **area_sqft** (0.28): En önemli faktör
2. **city** (0.19): Şehir etkisi
3. **neighborhood_score** (0.15): Mahalle kalitesi
4. **construction_quality** (0.12): Yapı kalitesi
5. **distance_to_center_km** (0.10): Merkeze mesafe
6. **age_years** (0.08): Emlak yaşı
7. **property_type** (0.06): Emlak tipi
8. **luxury_score** (0.05): Lüks özellikler

### Cross-Validation Sonuçları
- **5-Fold CV**: Tutarlı performans
- **Std Deviation**: Düşük varyans
- **Overfitting**: Minimal (validation vs training)

## İş Değeri

### Emlak Sektörü Faydaları
- **Otomatik Değerleme**: Hızlı fiyat tahmini
- **Pazar Analizi**: Yatırım kararı desteği
- **Risk Değerlendirmesi**: Overvaluation tespiti
- **Portföy Yönetimi**: Emlak portföy optimizasyonu

### Finans Sektörü Uygulamaları
- **Mortgage Değerlendirmesi**: Kredi risk analizi
- **Sigorta Primleri**: Emlak değer bazlı primler
- **Yatırım Danışmanlığı**: REİT ve emlak fonları
- **Varlık Yönetimi**: Institutional portfolio management

### Teknoloji Entegrasyonu
- **PropTech Platformları**: Online emlak değerleme
- **Mobile Apps**: Instant property valuation
- **IoT Integration**: Smart home value tracking
- **Blockchain**: Transparent property records

## Geliştirme Önerileri

1. **Gelişmiş Modeller**:
   - Neural Networks (Deep Learning)
   - LightGBM ve CatBoost
   - Bayesian Regression
   - Time series forecasting

2. **Ek Veri Kaynakları**:
   - Satellite imagery analysis
   - Economic indicators
   - Demographic data
   - Crime statistics
   - Transportation networks

3. **Real-time Features**:
   - Market trend APIs
   - Interest rate integration
   - Supply/demand metrics
   - Seasonal adjustments

4. **Advanced Analytics**:
   - Geospatial analysis (GIS)
   - Computer vision (property images)
   - Natural language processing (descriptions)
   - Sentiment analysis (reviews)

5. **Production Deployment**:
   - Docker containerization
   - Kubernetes orchestration
   - CI/CD pipelines
   - Monitoring ve alerting
   - A/B testing framework

## Örnek Kullanım Senaryoları

### Emlak Ajansı
```python
# Müşteri için fiyat analizi
property = get_property_details(property_id)
prediction = predict_price(property)
market_analysis = analyze_market(property)

# Müşteriye rapor
generate_valuation_report(prediction, market_analysis)
```

### Mortgage Şirketi
```python
# Kredi başvurusu değerlendirmesi
loan_amount = 400000
property_value = predict_price(property_data)
ltv_ratio = loan_amount / property_value['predicted_price']

# Risk analizi
if ltv_ratio > 0.8:
    risk_level = "High"
elif ltv_ratio > 0.6:
    risk_level = "Medium"
else:
    risk_level = "Low"
```

### Yatırım Platformu
```python
# Portföy optimizasyonu
properties = get_portfolio_properties()
valuations = []

for prop in properties:
    valuation = predict_price(prop)
    market_analysis = analyze_market(prop)
    valuations.append({
        'property_id': prop['id'],
        'current_value': valuation['predicted_price'],
        'investment_recommendation': market_analysis['investment_recommendation']
    })

# Yatırım stratejisi
optimize_portfolio(valuations)
```

