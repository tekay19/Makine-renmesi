# Proje 2: Sentiment Analysis ve Metin Sınıflandırma

Bu proje, doğal dil işleme (NLP) teknikleri kullanarak metin verilerini analiz eder, sentiment analysis ve kategori sınıflandırması yapar.

## Özellikler

### 📝 NLP Veri İşleme
- Metin temizleme ve normalizasyon
- Tokenization ve stop words kaldırma
- Stemming ve lemmatization
- TF-IDF ve N-gram feature extraction
- Anahtar kelime çıkarımı

### 🤖 Makine Öğrenmesi Modelleri
- **Multinomial Naive Bayes**: Sentiment analysis için
- **Support Vector Machine (SVM)**: Kategori sınıflandırması için
- **VADER Sentiment Analyzer**: Emotion scoring
- **TextBlob**: Polarity ve subjectivity analizi

### 🚀 FastAPI Servisi
- RESTful API endpoints
- Gerçek zamanlı metin analizi
- Kapsamlı sentiment ve kategori analizi
- Anahtar kelime çıkarımı servisi

## Kurulum

```bash
cd project2_sentiment_analysis
pip install -r requirements.txt
```

İlk çalıştırmada NLTK verileri otomatik olarak indirilecektir.

## Kullanım

### API Servisini Başlatma
```bash
python main.py
```

API şu adreste çalışacak: http://localhost:8002

### API Endpoints

#### 1. Sağlık Kontrolü
```bash
GET /health
```

#### 2. Sentiment Analysis
```bash
POST /analyze/sentiment
```

Örnek request body:
```json
{
  "text": "I love this new product! It's absolutely amazing and works perfectly.",
  "language": "en"
}
```

Örnek response:
```json
{
  "text": "I love this new product! It's absolutely amazing and works perfectly.",
  "sentiment": "positive",
  "confidence": 0.8542,
  "polarity": 0.625,
  "subjectivity": 0.9,
  "emotion_scores": {
    "neg": 0.0,
    "neu": 0.295,
    "pos": 0.705,
    "compound": 0.8316
  }
}
```

#### 3. Kategori Sınıflandırma
```bash
POST /analyze/category
```

#### 4. Kapsamlı Metin Analizi
```bash
POST /analyze/text
```

#### 5. Anahtar Kelime Çıkarımı
```bash
POST /extract/keywords?top_k=10
```

#### 6. Model Yeniden Eğitimi
```bash
POST /retrain
```

## Proje Yapısı

```
project2_sentiment_analysis/
├── main.py                    # Ana uygulama dosyası
├── requirements.txt           # Python bağımlılıkları
├── README.md                 # Proje dokümantasyonu
├── data/                     # Veri dosyaları
│   └── text_data.csv        # Metin verileri
└── models/                   # Eğitilmiş modeller
    └── text_analysis_models.pkl # Kaydedilmiş ML modelleri
```

## Teknik Detaylar

### NLP Pipeline
1. **Metin Temizleme**:
   - URL ve email kaldırma
   - Özel karakter temizleme
   - Küçük harf dönüşümü
   - Fazla boşluk kaldırma

2. **Tokenization ve Preprocessing**:
   - Word tokenization
   - Stop words kaldırma
   - Porter Stemming
   - Minimum kelime uzunluğu filtresi

3. **Feature Extraction**:
   - TF-IDF Vectorization
   - N-gram features (1-gram ve 2-gram)
   - Feature selection (max 5000 features)

### Makine Öğrenmesi Modelleri

#### Sentiment Analysis (Multinomial Naive Bayes)
- **Sınıflar**: Positive, Negative, Neutral
- **Özellikler**: TF-IDF vectors
- **Performans**: Cross-validation ile değerlendirme
- **Ek Analizler**: VADER emotion scores, TextBlob polarity

#### Kategori Sınıflandırma (SVM)
- **Kategoriler**: 
  - Technology
  - Sports
  - Politics
  - Entertainment
  - Business
  - Health
  - Science
  - Travel
- **Kernel**: Linear SVM
- **Probability**: Confidence scores için aktif

### Veri Seti
Proje otomatik olarak sentetik veri oluşturur:
- 8 farklı kategori
- 3000+ örnek metin
- Dengeli sentiment dağılımı
- Gerçekçi metin örnekleri

## API Kullanım Örnekleri

### Python ile API Kullanımı
```python
import requests

# Sentiment analysis
text_data = {
    "text": "This movie is absolutely fantastic! The acting is superb and the plot is engaging.",
    "language": "en"
}

response = requests.post("http://localhost:8002/analyze/sentiment", json=text_data)
result = response.json()

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
print(f"Polarity: {result['polarity']}")

# Kapsamlı analiz
response = requests.post("http://localhost:8002/analyze/text", json=text_data)
full_analysis = response.json()

print(f"Category: {full_analysis['category_analysis']['category']}")
print(f"Keywords: {full_analysis['keywords']}")
```

### cURL ile API Kullanımı
```bash
# Sentiment analysis
curl -X POST "http://localhost:8002/analyze/sentiment" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "I am really disappointed with this service. It was terrible!",
       "language": "en"
     }'

# Kategori analizi
curl -X POST "http://localhost:8002/analyze/category" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "The new smartphone features advanced AI capabilities and improved battery life."
     }'
```

### JavaScript ile API Kullanımı
```javascript
const analyzeText = async (text) => {
    const response = await fetch('http://localhost:8002/analyze/text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            language: 'en'
        })
    });
    
    const result = await response.json();
    return result;
};

// Kullanım
analyzeText("This is an amazing breakthrough in artificial intelligence!")
    .then(result => {
        console.log('Sentiment:', result.sentiment_analysis.sentiment);
        console.log('Category:', result.category_analysis.category);
        console.log('Keywords:', result.keywords);
    });
```

## Model Performansı

### Sentiment Analysis
- **Accuracy**: ~85-90%
- **Precision**: Yüksek positive/negative ayrımı
- **Recall**: Dengeli sınıf performansı
- **F1-Score**: Genel performans metriği

### Kategori Sınıflandırma
- **Accuracy**: ~80-85%
- **Multi-class Performance**: 8 kategori için dengeli
- **Confidence Scores**: Güvenilir olasılık tahminleri

### Ek Analizler
- **VADER Scores**: Emotion-based sentiment
- **TextBlob Polarity**: -1 (negative) to +1 (positive)
- **Subjectivity**: 0 (objective) to 1 (subjective)

## İş Değeri

### Sentiment Analysis Faydaları
- **Müşteri Geri Bildirimi**: Otomatik sentiment analizi
- **Sosyal Medya Monitoring**: Brand perception tracking
- **Ürün İncelemesi**: Review sentiment classification
- **Müşteri Hizmetleri**: Complaint priority detection

### Metin Sınıflandırma Faydaları
- **İçerik Organizasyonu**: Otomatik kategorizasyon
- **Haber Sınıflandırma**: News article categorization
- **Email Filtering**: Automatic email routing
- **Document Management**: Content-based filing

### Anahtar Kelime Çıkarımı
- **SEO Optimization**: Content keyword extraction
- **Trend Analysis**: Topic identification
- **Content Summarization**: Key concept extraction
- **Search Enhancement**: Relevant term identification

## Geliştirme Önerileri

1. **Gelişmiş Modeller**:
   - BERT/RoBERTa transformer modelleri
   - Deep learning approaches (LSTM, CNN)
   - Ensemble methods

2. **Çok Dilli Destek**:
   - Türkçe sentiment analysis
   - Language detection
   - Cross-lingual embeddings

3. **Real-time Processing**:
   - Streaming data processing
   - Kafka integration
   - Batch processing optimization

4. **Advanced Features**:
   - Named Entity Recognition (NER)
   - Aspect-based sentiment analysis
   - Emotion detection (joy, anger, fear, etc.)

5. **Monitoring ve Deployment**:
   - Model drift detection
   - A/B testing framework
   - Performance monitoring
   - Docker containerization

## Örnek Kullanım Senaryoları

### E-ticaret Platformu
```python
# Ürün yorumu analizi
review = "Bu ürün harika! Kalitesi çok iyi ve hızlı kargo."
result = analyze_sentiment(review)
# Pozitif yorumları highlight et, negatif yorumları müşteri hizmetlerine yönlendir
```

### Sosyal Medya Monitoring
```python
# Tweet analizi
tweet = "Yeni iPhone modeli gerçekten başarısız. Batarya ömrü çok kısa."
sentiment = analyze_sentiment(tweet)
category = analyze_category(tweet)
# Brand mention tracking ve crisis management
```

### Haber Sitesi
```python
# Haber makalesi sınıflandırma
article = "Yeni aşı araştırması umut verici sonuçlar gösteriyor..."
category = analyze_category(article)
# Otomatik kategorizasyon ve content routing
```

