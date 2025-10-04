# 🤖 Makine Öğrenmesi Yolculuğum

Merhaba! Ben Semih, ve bu repository'de makine öğrenmesi alanındaki öğrenme yolculuğumu ve geliştirdiğim projeleri paylaşıyorum. Her proje, farklı bir ML problemiyle uğraşırken öğrendiğim tekniklerin pratik uygulamaları.

## 🚀 Projelerim

### 🎯 1. Müşteri Segmentasyonu - İlk Büyük Projem
**Klasör**: [`project1_customer_segmentation/`](./project1_customer_segmentation/)  
**Kullandığım Teknolojiler**: K-means Clustering, Random Forest, FastAPI  
**Port**: 8001

Bu projeyi yaparken müşteri davranışlarını anlamaya çalıştım. K-means ile müşterileri gruplara ayırdım ve hangi müşterilerin ayrılma riski taşıdığını tahmin etmeye çalıştım. Gerçekten çok şey öğrendim!

### 📝 2. Duygu Analizi - NLP'ye Giriş Maceram
**Klasör**: [`project2_sentiment_analysis/`](./project2_sentiment_analysis/)  
**Kullandığım Teknolojiler**: NLP, TF-IDF, Naive Bayes, SVM, NLTK  
**Port**: 8002

Doğal dil işleme dünyasına ilk adımım bu proje oldu. Metinlerdeki duyguları anlamaya çalışmak gerçekten zorlu ama bir o kadar da heyecan vericiydi. 8 farklı kategori sınıflandırması yapabiliyor!

### 🏠 3. Ev Fiyat Tahmini - Regresyon Deneyimim  
**Klasör**: [`project3_price_prediction/`](./project3_price_prediction/)  
**Kullandığım Teknolojiler**: XGBoost, Random Forest, Ensemble Methods  
**Port**: 8003

Emlak fiyatlarını tahmin etmeye çalışırken ensemble metodları öğrendim. 4 farklı algoritmayı birleştirerek daha iyi sonuçlar elde etmeyi başardım.

### ⚽ 4. Futbol xG Modeli - Spor Tutkum + ML
**Dosya**: [`football_xg_model/training.py`](./football_xg_model/training.py)  
**Kullandığım Teknolojiler**: RandomForestClassifier, Pipeline, ROC-AUC

Futbol sevgimi makine öğrenmesiyle birleştirdiğim proje! Bir şutun gol olma olasılığını tahmin etmeye çalıştım. Oyuncu performanslarını sadece gol sayısıyla değil, pozisyon kalitesiyle de değerlendiriyorum.

### 🔍 5. Öneri Sistemi - E-ticaret Hayalim
**Dosyalar**: [`recommendation_system/main.py`](./recommendation_system/main.py), [`recommendation_system/main_commented.py`](./recommendation_system/main_commented.py)  
**Kullandığım Teknolojiler**: Collaborative Filtering, TF-IDF, Content-Based Filtering

Netflix ve Amazon'un nasıl öneri verdiğini merak ediyordum. Bu projeyle hibrit bir öneri sistemi yapmaya çalıştım. Hem kullanıcı davranışlarını hem de ürün özelliklerini kullanıyor.

## 🚀 Nasıl Çalıştırırsınız?

### Hepsini Denemek İsterseniz

```bash
# Önce repository'yi klonlayın
git clone https://github.com/tekay19/Makine-renmesi.git
cd Makine-renmesi

# Her proje için ayrı terminal açmanızı öneririm:

# Terminal 1 - Müşteri Segmentasyonu
cd project1_customer_segmentation
pip install -r requirements.txt
python main.py

# Terminal 2 - Duygu Analizi  
cd project2_sentiment_analysis
pip install -r requirements.txt
python main.py

# Terminal 3 - Fiyat Tahmini
cd project3_price_prediction
pip install -r requirements.txt
python main.py

# Terminal 4 - Futbol xG Modeli
cd football_xg_model
python training.py

# Terminal 5 - Öneri Sistemi
cd recommendation_system
python main.py
```

### API'leri Test Etmek İçin

- **Müşteri Segmentasyonu**: http://localhost:8001/docs
- **Duygu Analizi**: http://localhost:8002/docs  
- **Fiyat Tahmini**: http://localhost:8003/docs

Swagger UI'dan kolayca test edebilirsiniz!

## 📊 Projelerimin Detayları

### 🎯 Müşteri Segmentasyonu Maceramda Neler Yaptım

**Öğrendiğim şeyler**:
- K-means ile müşterileri nasıl gruplandıracağımı
- Random Forest ile churn prediction yapmayı
- FastAPI ile nasıl API oluşturacağımı
- Otomatik veri üretmeyi (gerçek veri bulamadığımda 😅)

**API'de neler var**:
- `POST /predict/churn` - Bu müşteri ayrılır mı?
- `POST /predict/segment` - Bu müşteri hangi grupta?
- `POST /analyze/customer` - Müşteri hakkında her şey

### 📝 Duygu Analizi Projesinde Keşfettiklerim

**Zorlandığım ama öğrendiğim konular**:
- Türkçe ve İngilizce metinleri nasıl işleyeceğim
- TF-IDF'in ne kadar güçlü olduğunu
- VADER ile emotion scoring yapmayı
- 8 farklı kategoriyi nasıl ayırt edeceğimi

**API'de neler deneyebilirsiniz**:
- `POST /analyze/sentiment` - Bu metin pozitif mi negatif mi?
- `POST /analyze/category` - Bu metin hangi kategoride?
- `POST /analyze/text` - Metinle ilgili her şeyi analiz et

### 🏠 Fiyat Tahmini Projesindeki Başarılarım

**En çok zorlandığım kısımlar**:
- Feature engineering (hangi özellikler önemli?)
- Ensemble metodları nasıl birleştireceğim
- XGBoost'u nasıl optimize edeceğim
- Pazar trendlerini nasıl modelleyeceğim

**API'de test edebilecekleriniz**:
- `POST /predict/price` - Bu ev ne kadar eder?
- `POST /analyze/market` - Pazar nasıl gidiyor?
- `GET /features/importance` - Hangi özellikler önemli?

### ⚽ Futbol xG Modelimde Öğrendiklerim

**Futbol sevgimle ML'i birleştirirken**:
- Pipeline'ların ne kadar kullanışlı olduğunu
- ROC-AUC'nin neden önemli olduğunu
- Oyuncu performansını nasıl objektif ölçeceğimi
- Şut kalitesinin sadece sonuçtan ibaret olmadığını

**Model çıktıları**:
- `goal_model.pkl` - Eğitilmiş modelim
- `player_stats.csv` - Oyuncu analizlerim

### 🔍 Öneri Sistemi Deneyimim

**En heyecan verici kısımları**:
- Collaborative filtering'in nasıl çalıştığını anlamak
- Content-based filtering ile soğuk başlangıç problemini çözmek
- İki yöntemi hibrit olarak birleştirmek
- TF-IDF ile ürün benzerliklerini bulmak

## 🛠️ Kullandığım Teknolojiler

### Öğrenme Sürecimde Keşfettiğim Araçlar

| Ne İçin Kullandım | Hangi Teknolojileri Öğrendim |
|-------------------|------------------------------|
| **Web API'leri** | FastAPI (çok sevdim!), Uvicorn |
| **ML Algoritmaları** | Scikit-learn, XGBoost, NLTK |
| **Veri İşleme** | Pandas (vazgeçilmez), NumPy, Joblib |
| **NLP** | TextBlob, VADER, TF-IDF |
| **Validation** | Pydantic (çok kullanışlı) |

### Modellerimin Performansları

| Hangi Proje | Nasıl Ölçtüm | Ne Kadar Başarılı |
|-------------|--------------|-------------------|
| Müşteri Segmentasyonu | Silhouette Score | 0.65-0.75 (fena değil!) |
| Churn Prediction | Accuracy | %85-90 (gurur duyuyorum) |
| Duygu Analizi | Accuracy | %85-90 (çok memnunum) |
| Fiyat Tahmini | R² Score | 0.85-0.92 (süper!) |
| Futbol xG | ROC-AUC | 0.80-0.85 (iyi gidiyor) |

## 📁 Repository Yapım

```
Makine-renmesi/
├── 📁 project1_customer_segmentation/  # İlk büyük projem
├── 📁 project2_sentiment_analysis/     # NLP maceramın başlangıcı
├── 📁 project3_price_prediction/       # Regresyon deneyimim
├── 📁 football_xg_model/               # Futbol tutkum + ML
├── 📁 recommendation_system/           # E-ticaret hayalim
├── 📄 README.md                        # Bu dosya
├── 📄 requirements.txt                 # Genel bağımlılıklar
└── 📄 .gitignore                       # Git kurallarım
```

## 🔧 Kurulum Rehberim

### Sisteminizde Olması Gerekenler
- Python 3.8+ (ben 3.9 kullanıyorum)
- 4GB+ RAM (bazen daha fazla gerekiyor)
- 2GB+ disk alanı

### Bağımlılıkları Kurmak İçin
```bash
pip install -r requirements.txt
```

Her projenin kendi requirements.txt dosyası var, onları da kontrol edin!

## 🧪 Nasıl Test Ederim

### Servislerin Çalışıp Çalışmadığını Kontrol Etmek
```bash
curl http://localhost:8001/health
curl http://localhost:8002/health  
curl http://localhost:8003/health
```

Hepsi "OK" dönerse her şey yolunda demektir!

## 🎯 Bu Projeleri Nerede Kullanabilirsiniz

### Müşteri Segmentasyonu
- CRM sistemlerinde müşteri yaşam döngüsü yönetimi
- Pazarlama kampanyalarını hedefleme
- Müşteri kaybını önleme stratejileri

### Duygu Analizi
- Sosyal medya marka takibi
- E-ticaret ürün yorumu analizi
- Müşteri hizmetlerinde otomatik ticket yönlendirme

### Fiyat Tahmini
- Emlak değerleme sistemleri
- Mortgage risk analizi
- Yatırım portföy optimizasyonu

### Futbol Analitik
- Oyuncu performans değerlendirmesi
- Transfer analizi ve scouting
- Takım stratejisi geliştirme

### Öneri Sistemi
- E-ticaret ürün önerileri
- İçerik platformları için kişiselleştirme
- Müşteri deneyimi optimizasyonu

## 📈 Gelecek Planlarım

### Yakın Zamanda Yapmak İstediklerim
- [ ] Docker ile containerization (öğrenmeye çalışıyorum)
- [ ] CI/CD pipeline kurmak (GitHub Actions ile)
- [ ] Monitoring ve logging sistemi eklemek
- [ ] Authentication sistemi yapmak

### Uzun Vadeli Hayallerim
- [ ] Kubernetes deployment öğrenmek
- [ ] Real-time streaming (Kafka ile)
- [ ] Deep learning modelleri denemek
- [ ] MLOps pipeline kurmak (MLflow)

## 🤝 Katkıda Bulunmak İsterseniz

Çok memnun olurum! Şöyle yapabilirsiniz:

1. Repository'yi fork edin
2. Kendi branch'inizi oluşturun (`git checkout -b feature/HarikaOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Harika bir özellik ekledim'`)
4. Branch'inizi push edin (`git push origin feature/HarikaOzellik`)
5. Pull Request gönderin

## 📞 Benimle İletişim

**Ben**: Semih Tekay  
**GitHub**: [@tekay19](https://github.com/tekay19)  
**Bu Repository**: [Makine-renmesi](https://github.com/tekay19/Makine-renmesi)

Sorularınız olursa çekinmeden sorun! Öğrenme yolculuğunda hep birlikte ilerleyelim.

---

⭐ Eğer projelerimi beğendiyseniz, yıldız vermeyi unutmayın! Bu beni çok mutlu eder ve motive eder 😊

**Not**: Bu projeler öğrenme amaçlı geliştirilmiştir. Production ortamında kullanmadan önce ek güvenlik ve optimizasyon çalışmaları yapılması gerekebilir.