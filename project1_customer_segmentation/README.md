# Proje 1: Müşteri Segmentasyonu ve Churn Prediction

Bu proje, müşteri verilerini analiz ederek müşteri segmentasyonu yapar ve churn (müşteri kaybı) tahminleri gerçekleştirir.

## Özellikler

### 🔍 Veri İşleme
- Otomatik örnek veri oluşturma
- Eksik değer işleme (imputation)
- Feature engineering (yeni özellik türetme)
- Kategorik değişken encoding
- Veri standardizasyonu

### 🎯 Makine Öğrenmesi Modelleri
- **K-means Clustering**: Müşteri segmentasyonu için
- **Random Forest**: Churn prediction için
- Model performans değerlendirmesi
- Feature importance analizi

### 🚀 FastAPI Servisi
- RESTful API endpoints
- Gerçek zamanlı tahmin servisi
- Kapsamlı müşteri analizi
- Model yeniden eğitimi

## Kurulum

```bash
cd project1_customer_segmentation
pip install -r requirements.txt
```

## Kullanım

### API Servisini Başlatma
```bash
python main.py
```

API şu adreste çalışacak: http://localhost:8001

### API Endpoints

#### 1. Sağlık Kontrolü
```bash
GET /health
```

#### 2. Churn Prediction
```bash
POST /predict/churn
```

Örnek request body:
```json
{
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
```

#### 3. Müşteri Segmentasyonu
```bash
POST /predict/segment
```

#### 4. Kapsamlı Müşteri Analizi
```bash
POST /analyze/customer
```

#### 5. Model Yeniden Eğitimi
```bash
POST /retrain
```

## Proje Yapısı

```
project1_customer_segmentation/
├── main.py                 # Ana uygulama dosyası
├── requirements.txt        # Python bağımlılıkları
├── README.md              # Proje dokümantasyonu
├── data/                  # Veri dosyaları
│   └── customer_data.csv  # Müşteri verileri
└── models/                # Eğitilmiş modeller
    └── customer_models.pkl # Kaydedilmiş ML modelleri
```

## Teknik Detaylar

### Veri İşleme Pipeline
1. **Veri Oluşturma**: Gerçekçi sentetik müşteri verisi
2. **Temizleme**: Eksik değer işleme ve outlier tespiti
3. **Feature Engineering**: 
   - `charges_per_month`: Aylık ortalama ücret
   - `is_senior`: Yaşlı müşteri flag'i
   - `high_value_customer`: Yüksek değerli müşteri
   - `long_tenure`: Uzun süreli müşteri

### Makine Öğrenmesi Modelleri

#### K-means Clustering
- **Amaç**: Müşteri segmentasyonu
- **Özellikler**: Yaş, tenure, ücretler, müşteri değeri
- **Segmentler**: 
  - Yüksek Değerli Müşteri
  - Orta Segment Müşteri
  - Düşük Değerli Müşteri
  - Yeni Müşteri

#### Random Forest Classifier
- **Amaç**: Churn prediction
- **Özellikler**: Demografik + hizmet + davranışsal veriler
- **Çıktı**: Churn olasılığı ve risk seviyesi

### Model Performansı
- **Silhouette Score**: Clustering kalitesi ölçümü
- **Classification Report**: Precision, Recall, F1-score
- **Feature Importance**: En önemli özelliklerin analizi

## API Kullanım Örnekleri

### Python ile API Kullanımı
```python
import requests

# Churn prediction
customer_data = {
    "customer_id": 1,
    "age": 45,
    "gender": "Female",
    "tenure_months": 12,
    "monthly_charges": 85.0,
    "total_charges": 1020.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "paperless_billing": "Yes"
}

response = requests.post("http://localhost:8001/predict/churn", json=customer_data)
print(response.json())
```

### cURL ile API Kullanımı
```bash
curl -X POST "http://localhost:8001/predict/churn" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": 1,
       "age": 45,
       "gender": "Female",
       "tenure_months": 12,
       "monthly_charges": 85.0,
       "total_charges": 1020.0,
       "contract_type": "Month-to-month",
       "payment_method": "Electronic check",
       "internet_service": "Fiber optic",
       "online_security": "No",
       "tech_support": "No",
       "streaming_tv": "Yes",
       "streaming_movies": "Yes",
       "paperless_billing": "Yes"
     }'
```

## İş Değeri

### Müşteri Segmentasyonu Faydaları
- **Hedefli Pazarlama**: Her segment için özelleştirilmiş kampanyalar
- **Kaynak Optimizasyonu**: Yüksek değerli müşterilere odaklanma
- **Kişiselleştirme**: Segment bazlı hizmet önerileri

### Churn Prediction Faydaları
- **Proaktif Müşteri Retention**: Risk altındaki müşterileri önceden tespit
- **Maliyet Azaltma**: Yeni müşteri kazanma maliyeti vs retention maliyeti
- **Gelir Koruma**: Değerli müşterilerin kaybını önleme

## Geliştirme Önerileri

1. **Daha Gelişmiş Modeller**: XGBoost, Neural Networks
2. **Real-time Streaming**: Kafka ile gerçek zamanlı veri işleme
3. **A/B Testing**: Model performansı karşılaştırması
4. **Monitoring**: Model drift ve performans takibi
5. **Explainable AI**: SHAP değerleri ile model açıklanabilirliği

