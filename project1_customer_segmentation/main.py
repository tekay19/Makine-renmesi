"""
Proje 1: Müşteri Segmentasyonu ve Churn Prediction
Bu proje müşteri verilerini analiz ederek segmentasyon yapar ve churn prediction modeli eğitir.

Özellikler:
- Veri işleme ve feature engineering
- K-means clustering ile müşteri segmentasyonu
- Random Forest ile churn prediction
- FastAPI ile RESTful servis
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
from sklearn.impute import SimpleImputer

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Dizin yapısı
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

class CustomerData(BaseModel):
    customer_id: int
    age: int
    gender: str
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    internet_service: str
    online_security: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    paperless_billing: str

class ChurnPrediction(BaseModel):
    customer_id: int
    churn_probability: float
    churn_prediction: bool
    segment: str
    risk_level: str

class SegmentationResult(BaseModel):
    customer_id: int
    segment: int
    segment_name: str
    characteristics: Dict[str, float]

class CustomerSegmentationModel:
    def __init__(self):
        self.kmeans_model = None
        self.churn_model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.segment_names = {
            0: "Yüksek Değerli Müşteri",
            1: "Orta Segment Müşteri", 
            2: "Düşük Değerli Müşteri",
            3: "Yeni Müşteri"
        }
        
    def generate_sample_data(self, n_customers=2000):
        """Örnek müşteri verisi oluştur"""
        np.random.seed(42)
        
        # Temel demografik veriler
        customer_ids = range(1, n_customers + 1)
        ages = np.random.normal(45, 15, n_customers).astype(int)
        ages = np.clip(ages, 18, 80)
        
        genders = np.random.choice(['Male', 'Female'], n_customers)
        
        # Hizmet verileri
        tenure_months = np.random.exponential(24, n_customers).astype(int)
        tenure_months = np.clip(tenure_months, 1, 72)
        
        # Ücret verileri (tenure ile korelasyonlu)
        base_charges = np.random.normal(65, 20, n_customers)
        monthly_charges = base_charges + np.random.normal(0, 10, n_customers)
        monthly_charges = np.clip(monthly_charges, 20, 120)
        
        total_charges = monthly_charges * tenure_months + np.random.normal(0, 100, n_customers)
        total_charges = np.maximum(total_charges, monthly_charges)
        
        # Kategorik veriler
        contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                        n_customers, p=[0.5, 0.3, 0.2])
        
        payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                         n_customers, p=[0.3, 0.2, 0.25, 0.25])
        
        internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                           n_customers, p=[0.4, 0.4, 0.2])
        
        # Ek hizmetler
        yes_no_services = ['Yes', 'No']
        online_security = np.random.choice(yes_no_services, n_customers, p=[0.3, 0.7])
        tech_support = np.random.choice(yes_no_services, n_customers, p=[0.3, 0.7])
        streaming_tv = np.random.choice(yes_no_services, n_customers, p=[0.4, 0.6])
        streaming_movies = np.random.choice(yes_no_services, n_customers, p=[0.4, 0.6])
        paperless_billing = np.random.choice(yes_no_services, n_customers, p=[0.6, 0.4])
        
        # Churn hesaplama (gerçekçi faktörler)
        churn_prob = np.zeros(n_customers)
        
        # Tenure etkisi (kısa tenure = yüksek churn)
        churn_prob += (24 - np.minimum(tenure_months, 24)) / 24 * 0.3
        
        # Contract type etkisi
        contract_effect = {'Month-to-month': 0.4, 'One year': 0.2, 'Two year': 0.1}
        for i, contract in enumerate(contract_types):
            churn_prob[i] += contract_effect[contract]
            
        # Ücret etkisi (yüksek ücret = yüksek churn)
        normalized_charges = (monthly_charges - monthly_charges.min()) / (monthly_charges.max() - monthly_charges.min())
        churn_prob += normalized_charges * 0.2
        
        # Yaş etkisi (genç müşteriler daha mobil)
        age_effect = (50 - np.minimum(ages, 50)) / 50 * 0.1
        churn_prob += age_effect
        
        # Random noise
        churn_prob += np.random.normal(0, 0.1, n_customers)
        churn_prob = np.clip(churn_prob, 0, 1)
        
        # Churn binary decision
        churn = (np.random.random(n_customers) < churn_prob).astype(int)
        
        # DataFrame oluştur
        data = pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'tenure_months': tenure_months,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_type': contract_types,
            'payment_method': payment_methods,
            'internet_service': internet_services,
            'online_security': online_security,
            'tech_support': tech_support,
            'streaming_tv': streaming_tv,
            'streaming_movies': streaming_movies,
            'paperless_billing': paperless_billing,
            'churn': churn
        })
        
        return data
    
    def preprocess_data(self, df):
        """Veri ön işleme"""
        # Kopyala
        data = df.copy()
        
        # Eksik değerleri doldur
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Numeric imputation
        numeric_imputer = SimpleImputer(strategy='median')
        data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
        
        # Categorical imputation
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
        
        # Feature engineering
        data['charges_per_month'] = data['total_charges'] / (data['tenure_months'] + 1)
        data['is_senior'] = (data['age'] >= 65).astype(int)
        data['high_value_customer'] = (data['monthly_charges'] > data['monthly_charges'].quantile(0.75)).astype(int)
        data['long_tenure'] = (data['tenure_months'] > 24).astype(int)
        
        # Kategorik değişkenleri encode et
        categorical_features = ['gender', 'contract_type', 'payment_method', 'internet_service',
                              'online_security', 'tech_support', 'streaming_tv', 'streaming_movies',
                              'paperless_billing']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                data[feature] = self.label_encoders[feature].fit_transform(data[feature])
            else:
                data[feature] = self.label_encoders[feature].transform(data[feature])
        
        return data
    
    def train_segmentation_model(self, df, n_clusters=4):
        """Müşteri segmentasyonu modeli eğit"""
        # Segmentasyon için özellikler
        segmentation_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                               'charges_per_month', 'high_value_customer', 'long_tenure']
        
        X_seg = df[segmentation_features].copy()
        
        # Standardize et
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_seg_scaled = self.scaler.fit_transform(X_seg)
        else:
            X_seg_scaled = self.scaler.transform(X_seg)
        
        # K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_seg_scaled)
        
        # Silhouette score hesapla
        silhouette_avg = silhouette_score(X_seg_scaled, clusters)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return clusters
    
    def train_churn_model(self, df):
        """Churn prediction modeli eğit"""
        # Churn prediction için özellikler
        feature_columns = ['age', 'gender', 'tenure_months', 'monthly_charges', 'total_charges',
                          'contract_type', 'payment_method', 'internet_service', 'online_security',
                          'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing',
                          'charges_per_month', 'is_senior', 'high_value_customer', 'long_tenure']
        
        X = df[feature_columns]
        y = df['churn']
        
        self.feature_columns = feature_columns
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Random Forest modeli
        self.churn_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.churn_model.fit(X_train, y_train)
        
        # Model performansı
        y_pred = self.churn_model.predict(X_test)
        print("Churn Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.churn_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.churn_model
    
    def predict_churn(self, customer_data):
        """Tek müşteri için churn prediction"""
        if self.churn_model is None:
            raise ValueError("Churn model not trained yet")
        
        # Veriyi işle
        df_single = pd.DataFrame([customer_data])
        df_processed = self.preprocess_data(df_single)
        
        # Prediction
        X = df_processed[self.feature_columns]
        churn_prob = self.churn_model.predict_proba(X)[0][1]
        churn_pred = self.churn_model.predict(X)[0]
        
        # Risk level
        if churn_prob > 0.7:
            risk_level = "Yüksek"
        elif churn_prob > 0.4:
            risk_level = "Orta"
        else:
            risk_level = "Düşük"
        
        return churn_prob, bool(churn_pred), risk_level
    
    def predict_segment(self, customer_data):
        """Tek müşteri için segment prediction"""
        if self.kmeans_model is None or self.scaler is None:
            raise ValueError("Segmentation model not trained yet")
        
        # Veriyi işle
        df_single = pd.DataFrame([customer_data])
        df_processed = self.preprocess_data(df_single)
        
        # Segmentasyon özellikleri
        segmentation_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                               'charges_per_month', 'high_value_customer', 'long_tenure']
        
        X_seg = df_processed[segmentation_features]
        X_seg_scaled = self.scaler.transform(X_seg)
        
        segment = self.kmeans_model.predict(X_seg_scaled)[0]
        segment_name = self.segment_names.get(segment, f"Segment {segment}")
        
        return segment, segment_name
    
    def save_models(self):
        """Modelleri kaydet"""
        models_data = {
            'kmeans_model': self.kmeans_model,
            'churn_model': self.churn_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'segment_names': self.segment_names
        }
        
        joblib.dump(models_data, MODELS_DIR / 'customer_models.pkl')
        print("Models saved successfully!")
    
    def load_models(self):
        """Modelleri yükle"""
        try:
            models_data = joblib.load(MODELS_DIR / 'customer_models.pkl')
            self.kmeans_model = models_data['kmeans_model']
            self.churn_model = models_data['churn_model']
            self.scaler = models_data['scaler']
            self.label_encoders = models_data['label_encoders']
            self.feature_columns = models_data['feature_columns']
            self.segment_names = models_data['segment_names']
            print("Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved models found.")
            return False

# Global model instance
model = CustomerSegmentationModel()

# FastAPI app
app = FastAPI(
    title="Müşteri Segmentasyonu ve Churn Prediction API",
    description="Müşteri segmentasyonu ve churn prediction için ML servisi",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında modelleri yükle veya eğit"""
    if not model.load_models():
        print("Training new models...")
        
        # Örnek veri oluştur
        df = model.generate_sample_data(2000)
        df.to_csv(DATA_DIR / 'customer_data.csv', index=False)
        
        # Veriyi işle
        df_processed = model.preprocess_data(df)
        
        # Modelleri eğit
        clusters = model.train_segmentation_model(df_processed)
        df_processed['segment'] = clusters
        
        model.train_churn_model(df_processed)
        
        # Modelleri kaydet
        model.save_models()
        
        print("Models trained and saved!")

@app.get("/")
async def root():
    return {
        "message": "Müşteri Segmentasyonu ve Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_churn": "POST /predict/churn",
            "predict_segment": "POST /predict/segment",
            "customer_analysis": "POST /analyze/customer",
            "retrain": "POST /retrain",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": model.churn_model is not None and model.kmeans_model is not None
    }

@app.post("/predict/churn", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerData):
    """Müşteri churn prediction"""
    try:
        customer_dict = customer.dict()
        churn_prob, churn_pred, risk_level = model.predict_churn(customer_dict)
        segment, segment_name = model.predict_segment(customer_dict)
        
        return ChurnPrediction(
            customer_id=customer.customer_id,
            churn_probability=round(churn_prob, 4),
            churn_prediction=churn_pred,
            segment=segment_name,
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/segment", response_model=SegmentationResult)
async def predict_segment(customer: CustomerData):
    """Müşteri segmentasyonu"""
    try:
        customer_dict = customer.dict()
        segment, segment_name = model.predict_segment(customer_dict)
        
        # Segment karakteristikleri (örnek)
        characteristics = {
            "monthly_charges": customer.monthly_charges,
            "tenure_months": customer.tenure_months,
            "total_charges": customer.total_charges
        }
        
        return SegmentationResult(
            customer_id=customer.customer_id,
            segment=segment,
            segment_name=segment_name,
            characteristics=characteristics
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/customer")
async def analyze_customer(customer: CustomerData):
    """Kapsamlı müşteri analizi"""
    try:
        customer_dict = customer.dict()
        
        # Churn prediction
        churn_prob, churn_pred, risk_level = model.predict_churn(customer_dict)
        
        # Segmentation
        segment, segment_name = model.predict_segment(customer_dict)
        
        # Öneriler
        recommendations = []
        
        if churn_prob > 0.5:
            recommendations.append("Müşteri retention kampanyası uygulayın")
            recommendations.append("Kişiselleştirilmiş indirim teklifi sunun")
        
        if customer.tenure_months < 12:
            recommendations.append("Yeni müşteri onboarding programına dahil edin")
        
        if customer.monthly_charges > 80:
            recommendations.append("Premium müşteri hizmetleri sunun")
        
        return {
            "customer_id": customer.customer_id,
            "churn_analysis": {
                "probability": round(churn_prob, 4),
                "prediction": churn_pred,
                "risk_level": risk_level
            },
            "segmentation": {
                "segment": segment,
                "segment_name": segment_name
            },
            "recommendations": recommendations,
            "customer_value": "Yüksek" if customer.monthly_charges > 70 else "Orta" if customer.monthly_charges > 40 else "Düşük"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
async def retrain_models():
    """Modelleri yeniden eğit"""
    try:
        # Yeni veri oluştur
        df = model.generate_sample_data(2500)
        df.to_csv(DATA_DIR / 'customer_data_new.csv', index=False)
        
        # Veriyi işle
        df_processed = model.preprocess_data(df)
        
        # Modelleri yeniden eğit
        clusters = model.train_segmentation_model(df_processed)
        df_processed['segment'] = clusters
        
        model.train_churn_model(df_processed)
        
        # Modelleri kaydet
        model.save_models()
        
        return {"status": "success", "message": "Models retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
