# 🤖 Makine Öğrenmesi Projeleri Koleksiyonu

Bu repository, kapsamlı makine öğrenmesi projelerini ve futbol analitik çözümlerini içeren profesyonel bir koleksiyondur. Her proje, gerçek dünya problemlerine yönelik çözümler sunar ve modern ML teknolojilerini kullanır.

## 🚀 Projeler

### 🎯 1. Müşteri Segmentasyonu ve Churn Prediction
**Dizin**: [`project1_customer_segmentation/`](./project1_customer_segmentation/)  
**Teknolojiler**: K-means Clustering, Random Forest, FastAPI  
**Port**: 8001

Müşteri davranış analizi ve churn tahmini için geliştirilmiş kapsamlı sistem.

### 📝 2. Sentiment Analysis ve Metin Sınıflandırma  
**Dizin**: [`project2_sentiment_analysis/`](./project2_sentiment_analysis/)  
**Teknolojiler**: NLP, TF-IDF, Naive Bayes, SVM, NLTK  
**Port**: 8002

Gelişmiş doğal dil işleme ve duygu analizi sistemi.

### 🏠 3. Fiyat Tahmini ve Regresyon Analizi
**Dizin**: [`project3_price_prediction/`](./project3_price_prediction/)  
**Teknolojiler**: XGBoost, Random Forest, Ensemble Methods  
**Port**: 8003

Emlak fiyat tahmini ve pazar analizi sistemi.

### ⚽ 4. Futbol xG (Expected Goals) Modeli
**Dosya**: [`training.py`](./training.py)  
**Teknolojiler**: RandomForestClassifier, Pipeline, ROC-AUC

Futbol maçlarındaki şutların gol olma olasılığını tahmin eden gelişmiş model.

### 🔍 5. Öneri Sistemi (Recommendation Engine)
**Dosyalar**: [`main.py`](./main.py), [`main_commented.py`](./main_commented.py)  
**Teknolojiler**: Collaborative Filtering, TF-IDF, Content-Based Filtering

Hibrit öneri sistemi implementasyonu.

## 🚀 Hızlı Başlangıç

### Tüm Projeleri Çalıştırma

```bash
# Repository'yi klonlayın
git clone https://github.com/tekay19/Makine-renmesi.git
cd Makine-renmesi

# Her proje için ayrı terminal açın:

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

# Terminal 4 - Futbol xG Modeli
python training.py

# Terminal 5 - Öneri Sistemi
python main.py
```

### API Erişim Adresleri

- **Müşteri Segmentasyonu**: http://localhost:8001/docs
- **Sentiment Analysis**: http://localhost:8002/docs  
- **Fiyat Tahmini**: http://localhost:8003/docs

## 📊 Proje Detayları

### 🎯 Müşteri Segmentasyonu ve Churn Prediction

**Özellikler**:
- K-means ile müşteri segmentasyonu
- Random Forest ile churn tahmini
- Otomatik veri oluşturma
- FastAPI ile RESTful servis

**API Endpoints**:
- `POST /predict/churn` - Churn tahmini
- `POST /predict/segment` - Müşteri segmentasyonu
- `POST /analyze/customer` - Kapsamlı müşteri analizi

### 📝 Sentiment Analysis ve Metin Sınıflandırma

**Özellikler**:
- Çok dilli sentiment analizi
- 8 kategoride metin sınıflandırması
- Anahtar kelime çıkarımı
- VADER emotion scoring

**API Endpoints**:
- `POST /analyze/sentiment` - Sentiment analizi
- `POST /analyze/category` - Kategori sınıflandırması
- `POST /analyze/text` - Kapsamlı metin analizi

### 🏠 Fiyat Tahmini ve Regresyon Analizi

**Özellikler**:
- XGBoost tabanlı fiyat tahmini
- Ensemble model (4 farklı algoritma)
- Pazar analizi ve yatırım önerileri
- Feature importance analizi

**API Endpoints**:
- `POST /predict/price` - Fiyat tahmini
- `POST /analyze/market` - Pazar analizi
- `GET /features/importance` - Feature importance

### ⚽ Futbol xG (Expected Goals) Modeli

**Özellikler**:
- RandomForestClassifier ile şut analizi
- Pipeline tabanlı veri işleme
- Oyuncu performans değerlendirmesi
- ROC-AUC metriği ile model değerlendirme

**Çıktılar**:
- `goal_model.pkl` - Eğitilmiş model
- `player_stats.csv` - Oyuncu istatistikleri

### 🔍 Öneri Sistemi (Recommendation Engine)

**Özellikler**:
- Collaborative Filtering
- Content-Based Filtering
- TF-IDF tabanlı benzerlik
- Hibrit öneri algoritması

## 🛠️ Teknik Özellikler

### Kullanılan Teknolojiler

| Kategori | Teknolojiler |
|----------|-------------|
| **Web Framework** | FastAPI, Uvicorn |
| **ML Kütüphaneleri** | Scikit-learn, XGBoost, NLTK |
| **Veri İşleme** | Pandas, NumPy, Joblib |
| **NLP** | TextBlob, VADER, TF-IDF |
| **Validation** | Pydantic |

### Model Performansları

| Proje | Metrik | Performans |
|-------|--------|------------|
| Müşteri Segmentasyonu | Silhouette Score | 0.65-0.75 |
| Churn Prediction | Accuracy | 85-90% |
| Sentiment Analysis | Accuracy | 85-90% |
| Fiyat Tahmini | R² Score | 0.85-0.92 |
| Futbol xG | ROC-AUC | 0.80-0.85 |

## 📁 Proje Yapısı

```
Makine-renmesi/
├── 📁 project1_customer_segmentation/
│   ├── 📄 main.py
│   ├── 📄 README.md
│   ├── 📄 requirements.txt
│   ├── 📁 data/
│   └── 📁 models/
├── 📁 project2_sentiment_analysis/
│   ├── 📄 main.py
│   ├── 📄 README.md
│   └── 📄 requirements.txt
├── 📁 project3_price_prediction/
│   ├── 📄 main.py
│   ├── 📄 README.md
│   └── 📄 requirements.txt
├── 📁 football_xg_model/
│   ├── 📄 training.py
│   └── 📄 README.md
├── 📁 recommendation_system/
│   ├── 📄 main.py
│   ├── 📄 main_commented.py
│   ├── 📁 data/
│   ├── 📁 artifacts/
│   └── 📄 README.md
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 .gitignore
```

## 🔧 Kurulum ve Gereksinimler

### Sistem Gereksinimleri
- Python 3.8+
- 4GB+ RAM
- 2GB+ disk alanı

### Genel Bağımlılıklar
```bash
pip install -r requirements.txt
```

### Proje Özel Bağımlılıklar
Her proje kendi `requirements.txt` dosyasına sahiptir.

## 🧪 Test Etme

### Sağlık Kontrolü
```bash
curl http://localhost:8001/health
curl http://localhost:8002/health  
curl http://localhost:8003/health
```

### Örnek API Çağrıları
Detaylı örnekler için her projenin kendi README dosyasını inceleyiniz.

## 🎯 İş Değeri ve Kullanım Alanları

### Müşteri Segmentasyonu
- CRM sistemleri için müşteri lifecycle yönetimi
- Hedefli pazarlama kampanyaları
- Churn önleme stratejileri

### Sentiment Analysis
- Sosyal medya brand monitoring
- E-ticaret ürün yorumu analizi
- Müşteri hizmetleri otomatik ticket routing

### Fiyat Tahmini
- Emlak otomatik değerleme sistemleri
- Mortgage risk analizi
- Yatırım portföy optimizasyonu

### Futbol Analitik
- Oyuncu performans değerlendirmesi
- Transfer analizi
- Takım stratejisi geliştirme

### Öneri Sistemi
- E-ticaret ürün önerileri
- İçerik platformları için kişiselleştirme
- Müşteri deneyimi optimizasyonu

## 📈 Geliştirme Roadmap

### Kısa Vadeli İyileştirmeler
- [ ] Docker containerization
- [ ] CI/CD pipeline kurulumu
- [ ] Monitoring ve logging sistemi
- [ ] Authentication ve authorization

### Uzun Vadeli Geliştirmeler
- [ ] Kubernetes deployment
- [ ] Real-time streaming (Kafka)
- [ ] Deep learning modelleri
- [ ] MLOps pipeline (MLflow)

## 🤝 Katkıda Bulunma

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasını inceleyiniz.

## 📞 İletişim

**Geliştirici**: Semih Tekay  
**GitHub**: [@tekay19](https://github.com/tekay19)  
**Repository**: [Makine-renmesi](https://github.com/tekay19/Makine-renmesi)

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!