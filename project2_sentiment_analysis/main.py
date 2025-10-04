"""
Proje 2: Sentiment Analysis ve Metin Sınıflandırma
Bu proje metin verilerini analiz ederek sentiment analysis ve kategori sınıflandırması yapar.

Özellikler:
- NLP veri işleme ve temizleme
- TF-IDF ve Word2Vec feature extraction
- Naive Bayes ve SVM modelleri
- FastAPI ile metin analizi servisi
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from textblob import TextBlob

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# NLTK downloads
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Dizin yapısı
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

class TextInput(BaseModel):
    text: str
    language: Optional[str] = "en"

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    polarity: float
    subjectivity: float
    emotion_scores: Dict[str, float]

class CategoryResult(BaseModel):
    text: str
    category: str
    confidence: float
    all_probabilities: Dict[str, float]

class TextAnalysisResult(BaseModel):
    text: str
    sentiment_analysis: SentimentResult
    category_analysis: CategoryResult
    text_stats: Dict[str, int]
    keywords: List[str]

class TextAnalysisModel:
    def __init__(self):
        self.sentiment_model = None
        self.category_model = None
        self.tfidf_vectorizer = None
        self.stemmer = PorterStemmer()
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Kategori etiketleri
        self.categories = [
            'technology', 'sports', 'politics', 'entertainment', 
            'business', 'health', 'science', 'travel'
        ]
        
    def generate_sample_data(self, n_samples=3000):
        """Örnek metin verisi oluştur"""
        np.random.seed(42)
        
        # Kategori bazlı örnek metinler
        sample_texts = {
            'technology': [
                "The new smartphone features advanced AI capabilities and improved battery life.",
                "Machine learning algorithms are revolutionizing data analysis in various industries.",
                "Cloud computing services provide scalable solutions for modern businesses.",
                "Artificial intelligence is transforming the way we interact with technology.",
                "The latest software update includes enhanced security features and bug fixes.",
                "Blockchain technology offers transparent and secure transaction processing.",
                "Virtual reality applications are expanding beyond gaming into education and training.",
                "Internet of Things devices are creating smarter homes and cities.",
                "Cybersecurity measures are crucial for protecting sensitive digital information.",
                "Quantum computing promises to solve complex problems at unprecedented speeds."
            ],
            'sports': [
                "The football team won the championship with an outstanding performance.",
                "Basketball players demonstrated exceptional skills during the tournament.",
                "The tennis match was intense with both players showing great determination.",
                "Olympic athletes broke several world records in swimming competitions.",
                "The soccer team's strategy led them to victory in the final match.",
                "Baseball season started with exciting games and promising new players.",
                "Marathon runners showed incredible endurance throughout the race.",
                "The hockey team's defense was impenetrable during the playoffs.",
                "Golf tournament featured challenging courses and skilled professionals.",
                "Boxing match ended with a surprising knockout in the final round."
            ],
            'politics': [
                "The new policy aims to improve healthcare access for all citizens.",
                "Election results show significant changes in voter preferences.",
                "Government officials announced new economic stimulus measures.",
                "International relations require careful diplomatic negotiations.",
                "Legislative reforms focus on environmental protection and sustainability.",
                "Political debates highlight different approaches to social issues.",
                "Voting rights legislation ensures fair and accessible elections.",
                "Foreign policy decisions impact global trade relationships.",
                "Constitutional amendments require broad public support and approval.",
                "Political campaigns emphasize transparency and accountability."
            ],
            'entertainment': [
                "The movie received critical acclaim for its outstanding cinematography.",
                "Music festival featured diverse artists from around the world.",
                "Television series finale drew millions of viewers worldwide.",
                "Theater production showcased exceptional acting and stage design.",
                "Celebrity interviews revealed interesting behind-the-scenes stories.",
                "Concert tour announcement excited fans across multiple cities.",
                "Film festival highlighted independent movies and emerging directors.",
                "Gaming industry continues to innovate with immersive experiences.",
                "Streaming platforms offer extensive libraries of original content.",
                "Award ceremony celebrated achievements in various entertainment fields."
            ],
            'business': [
                "Company quarterly earnings exceeded analyst expectations significantly.",
                "Startup secured major funding for innovative product development.",
                "Market analysis reveals growing trends in sustainable investing.",
                "Corporate merger creates opportunities for expanded market reach.",
                "Business strategy focuses on digital transformation and efficiency.",
                "Economic indicators suggest positive growth in key sectors.",
                "Investment portfolio diversification reduces financial risk exposure.",
                "Supply chain optimization improves operational efficiency and costs.",
                "Customer satisfaction surveys drive product improvement initiatives.",
                "Financial planning helps businesses navigate uncertain market conditions."
            ],
            'health': [
                "Medical research breakthrough offers hope for cancer treatment.",
                "Healthy lifestyle choices significantly reduce disease risk factors.",
                "Mental health awareness campaigns promote well-being and support.",
                "Vaccination programs protect communities from infectious diseases.",
                "Nutrition studies reveal benefits of balanced diet and exercise.",
                "Healthcare technology improves patient care and treatment outcomes.",
                "Preventive medicine focuses on early detection and intervention.",
                "Medical professionals provide essential services during health crises.",
                "Pharmaceutical developments lead to more effective medications.",
                "Public health initiatives address widespread health challenges."
            ],
            'science': [
                "Scientific discovery advances our understanding of the universe.",
                "Research findings contribute to climate change mitigation strategies.",
                "Laboratory experiments validate theoretical physics predictions.",
                "Space exploration missions reveal fascinating planetary information.",
                "Environmental studies guide conservation and protection efforts.",
                "Genetic research opens new possibilities for medical treatments.",
                "Archaeological discoveries provide insights into ancient civilizations.",
                "Marine biology research explores ocean ecosystems and biodiversity.",
                "Chemistry innovations lead to sustainable material development.",
                "Astronomy observations expand knowledge of distant galaxies."
            ],
            'travel': [
                "Vacation destination offers beautiful beaches and cultural attractions.",
                "Travel guide recommends must-see landmarks and local experiences.",
                "Adventure tourism provides thrilling outdoor activities and exploration.",
                "Hotel accommodations feature luxury amenities and excellent service.",
                "Flight booking platforms offer competitive prices and convenient schedules.",
                "Cultural exchange programs promote international understanding.",
                "Backpacking journey creates memorable experiences and friendships.",
                "Cruise ship vacation includes entertainment and exotic destinations.",
                "Travel insurance provides protection for unexpected situations.",
                "Local cuisine exploration enhances cultural travel experiences."
            ]
        }
        
        # Sentiment kelimeleri
        positive_words = ['excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'outstanding', 'brilliant', 'perfect', 'incredible', 'superb']
        negative_words = ['terrible', 'awful', 'horrible', 'disappointing', 'bad', 'poor', 'worst', 'failed', 'disaster', 'pathetic']
        neutral_words = ['okay', 'average', 'normal', 'standard', 'typical', 'regular', 'common', 'usual', 'ordinary', 'moderate']
        
        texts = []
        categories = []
        sentiments = []
        
        for category, base_texts in sample_texts.items():
            # Her kategori için örnekler oluştur
            category_samples = n_samples // len(sample_texts)
            
            for _ in range(category_samples):
                base_text = np.random.choice(base_texts)
                
                # Sentiment ekle
                sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
                
                if sentiment_type == 'positive':
                    modifier = np.random.choice(positive_words)
                    text = base_text.replace(base_text.split()[0], f"{modifier} {base_text.split()[0]}")
                    sentiment = 'positive'
                elif sentiment_type == 'negative':
                    modifier = np.random.choice(negative_words)
                    text = base_text.replace(base_text.split()[0], f"{modifier} {base_text.split()[0]}")
                    sentiment = 'negative'
                else:
                    modifier = np.random.choice(neutral_words)
                    text = base_text.replace(base_text.split()[0], f"{modifier} {base_text.split()[0]}")
                    sentiment = 'neutral'
                
                # Varyasyon ekle
                if np.random.random() > 0.5:
                    text += f" This is {np.random.choice(['really', 'very', 'quite', 'extremely'])} {np.random.choice(['interesting', 'important', 'relevant', 'significant'])}."
                
                texts.append(text)
                categories.append(category)
                sentiments.append(sentiment)
        
        # DataFrame oluştur
        df = pd.DataFrame({
            'text': texts,
            'category': categories,
            'sentiment': sentiments
        })
        
        return df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    def clean_text(self, text):
        """Metin temizleme"""
        # Küçük harfe çevir
        text = text.lower()
        
        # URL'leri kaldır
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Email adreslerini kaldır
        text = re.sub(r'\S+@\S+', '', text)
        
        # Özel karakterleri kaldır (sadece harf, rakam ve boşluk bırak)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Fazla boşlukları kaldır
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text):
        """Gelişmiş metin ön işleme"""
        # Temizle
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Stop words ve kısa kelimeleri kaldır
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_features(self, texts, fit_vectorizer=False):
        """Metin özelliklerini çıkar"""
        # Metinleri ön işle
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        if fit_vectorizer or self.tfidf_vectorizer is None:
            # TF-IDF vectorizer oluştur
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            features = self.tfidf_vectorizer.fit_transform(processed_texts)
        else:
            features = self.tfidf_vectorizer.transform(processed_texts)
        
        return features
    
    def train_sentiment_model(self, df):
        """Sentiment analysis modeli eğit"""
        # Özellik çıkarımı
        X = self.extract_features(df['text'].tolist(), fit_vectorizer=True)
        y = df['sentiment']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model eğitimi (Multinomial Naive Bayes)
        self.sentiment_model = MultinomialNB(alpha=1.0)
        self.sentiment_model.fit(X_train, y_train)
        
        # Model performansı
        y_pred = self.sentiment_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Sentiment Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.sentiment_model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.sentiment_model
    
    def train_category_model(self, df):
        """Kategori sınıflandırma modeli eğit"""
        # Aynı TF-IDF vectorizer'ı kullan
        X = self.extract_features(df['text'].tolist())
        y = df['category']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model eğitimi (SVM)
        self.category_model = SVC(
            kernel='linear',
            probability=True,
            random_state=42
        )
        self.category_model.fit(X_train, y_train)
        
        # Model performansı
        y_pred = self.category_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Category Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.category_model
    
    def predict_sentiment(self, text):
        """Sentiment prediction"""
        if self.sentiment_model is None or self.tfidf_vectorizer is None:
            raise ValueError("Sentiment model not trained yet")
        
        # TF-IDF features
        features = self.extract_features([text])
        
        # Prediction
        sentiment_pred = self.sentiment_model.predict(features)[0]
        sentiment_proba = self.sentiment_model.predict_proba(features)[0]
        
        # Confidence (en yüksek olasılık)
        confidence = max(sentiment_proba)
        
        # VADER sentiment analysis
        vader_scores = self.sia.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return {
            'sentiment': sentiment_pred,
            'confidence': confidence,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'emotion_scores': vader_scores
        }
    
    def predict_category(self, text):
        """Kategori prediction"""
        if self.category_model is None or self.tfidf_vectorizer is None:
            raise ValueError("Category model not trained yet")
        
        # TF-IDF features
        features = self.extract_features([text])
        
        # Prediction
        category_pred = self.category_model.predict(features)[0]
        category_proba = self.category_model.predict_proba(features)[0]
        
        # Tüm kategoriler için olasılıklar
        categories = self.category_model.classes_
        all_probabilities = dict(zip(categories, category_proba))
        
        # Confidence
        confidence = max(category_proba)
        
        return {
            'category': category_pred,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
    
    def extract_keywords(self, text, top_k=5):
        """Anahtar kelime çıkarımı"""
        if self.tfidf_vectorizer is None:
            return []
        
        # TF-IDF features
        features = self.extract_features([text])
        
        # Feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # TF-IDF scores
        tfidf_scores = features.toarray()[0]
        
        # En yüksek skorlu kelimeleri bul
        top_indices = tfidf_scores.argsort()[-top_k:][::-1]
        keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
        
        return keywords
    
    def get_text_statistics(self, text):
        """Metin istatistikleri"""
        words = word_tokenize(text.lower())
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'character_count': len(text),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'unique_words': len(set(words))
        }
    
    def save_models(self):
        """Modelleri kaydet"""
        models_data = {
            'sentiment_model': self.sentiment_model,
            'category_model': self.category_model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'categories': self.categories
        }
        
        joblib.dump(models_data, MODELS_DIR / 'text_analysis_models.pkl')
        print("Models saved successfully!")
    
    def load_models(self):
        """Modelleri yükle"""
        try:
            models_data = joblib.load(MODELS_DIR / 'text_analysis_models.pkl')
            self.sentiment_model = models_data['sentiment_model']
            self.category_model = models_data['category_model']
            self.tfidf_vectorizer = models_data['tfidf_vectorizer']
            self.categories = models_data['categories']
            print("Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved models found.")
            return False

# Global model instance
model = TextAnalysisModel()

# FastAPI app
app = FastAPI(
    title="Sentiment Analysis ve Metin Sınıflandırma API",
    description="NLP tabanlı metin analizi servisi",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında modelleri yükle veya eğit"""
    if not model.load_models():
        print("Training new models...")
        
        # Örnek veri oluştur
        df = model.generate_sample_data(3000)
        df.to_csv(DATA_DIR / 'text_data.csv', index=False)
        
        # Modelleri eğit
        model.train_sentiment_model(df)
        model.train_category_model(df)
        
        # Modelleri kaydet
        model.save_models()
        
        print("Models trained and saved!")

@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis ve Metin Sınıflandırma API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze_sentiment": "POST /analyze/sentiment",
            "analyze_category": "POST /analyze/category", 
            "analyze_text": "POST /analyze/text",
            "extract_keywords": "POST /extract/keywords",
            "retrain": "POST /retrain",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": model.sentiment_model is not None and model.category_model is not None
    }

@app.post("/analyze/sentiment", response_model=SentimentResult)
async def analyze_sentiment(text_input: TextInput):
    """Sentiment analysis"""
    try:
        result = model.predict_sentiment(text_input.text)
        
        return SentimentResult(
            text=text_input.text,
            sentiment=result['sentiment'],
            confidence=round(result['confidence'], 4),
            polarity=round(result['polarity'], 4),
            subjectivity=round(result['subjectivity'], 4),
            emotion_scores={k: round(v, 4) for k, v in result['emotion_scores'].items()}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/category", response_model=CategoryResult)
async def analyze_category(text_input: TextInput):
    """Kategori sınıflandırma"""
    try:
        result = model.predict_category(text_input.text)
        
        return CategoryResult(
            text=text_input.text,
            category=result['category'],
            confidence=round(result['confidence'], 4),
            all_probabilities={k: round(v, 4) for k, v in result['all_probabilities'].items()}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/text", response_model=TextAnalysisResult)
async def analyze_text(text_input: TextInput):
    """Kapsamlı metin analizi"""
    try:
        # Sentiment analysis
        sentiment_result = model.predict_sentiment(text_input.text)
        
        # Category analysis
        category_result = model.predict_category(text_input.text)
        
        # Text statistics
        text_stats = model.get_text_statistics(text_input.text)
        
        # Keywords
        keywords = model.extract_keywords(text_input.text)
        
        return TextAnalysisResult(
            text=text_input.text,
            sentiment_analysis=SentimentResult(
                text=text_input.text,
                sentiment=sentiment_result['sentiment'],
                confidence=round(sentiment_result['confidence'], 4),
                polarity=round(sentiment_result['polarity'], 4),
                subjectivity=round(sentiment_result['subjectivity'], 4),
                emotion_scores={k: round(v, 4) for k, v in sentiment_result['emotion_scores'].items()}
            ),
            category_analysis=CategoryResult(
                text=text_input.text,
                category=category_result['category'],
                confidence=round(category_result['confidence'], 4),
                all_probabilities={k: round(v, 4) for k, v in category_result['all_probabilities'].items()}
            ),
            text_stats=text_stats,
            keywords=keywords
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/extract/keywords")
async def extract_keywords(text_input: TextInput, top_k: int = 10):
    """Anahtar kelime çıkarımı"""
    try:
        keywords = model.extract_keywords(text_input.text, top_k)
        text_stats = model.get_text_statistics(text_input.text)
        
        return {
            "text": text_input.text,
            "keywords": keywords,
            "text_statistics": text_stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
async def retrain_models():
    """Modelleri yeniden eğit"""
    try:
        # Yeni veri oluştur
        df = model.generate_sample_data(4000)
        df.to_csv(DATA_DIR / 'text_data_new.csv', index=False)
        
        # Modelleri yeniden eğit
        model.train_sentiment_model(df)
        model.train_category_model(df)
        
        # Modelleri kaydet
        model.save_models()
        
        return {"status": "success", "message": "Models retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

