"""
Proje 3: Fiyat Tahmini ve Regresyon Analizi
Bu proje çeşitli özellikler kullanarak fiyat tahmini yapar ve regresyon analizi gerçekleştirir.

Özellikler:
- Kapsamlı feature engineering
- XGBoost ve Linear Regression modelleri
- Model ensemble ve stacking
- FastAPI ile fiyat tahmin servisi
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Feature Engineering
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Dizin yapısı
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

class PropertyInput(BaseModel):
    # Temel özellikler
    area_sqft: float
    bedrooms: int
    bathrooms: float
    floors: int
    age_years: int
    
    # Lokasyon özellikleri
    city: str
    district: str
    neighborhood_score: float  # 1-10 arası
    
    # Yapısal özellikler
    property_type: str  # house, apartment, villa, etc.
    construction_quality: str  # low, medium, high, luxury
    heating_type: str
    parking_spaces: int
    
    # Çevresel özellikler
    distance_to_center_km: float
    distance_to_metro_km: float
    distance_to_school_km: float
    distance_to_hospital_km: float
    distance_to_mall_km: float
    
    # Ek özellikler
    has_garden: bool
    has_balcony: bool
    has_elevator: bool
    has_security: bool
    has_gym: bool
    has_pool: bool
    
    # Pazar özellikleri
    market_trend: str  # rising, stable, declining
    season: str  # spring, summer, autumn, winter

class PricePrediction(BaseModel):
    predicted_price: float
    confidence_interval: Tuple[float, float]
    model_confidence: float
    price_per_sqft: float
    market_analysis: Dict[str, str]
    feature_importance: Dict[str, float]

class MarketAnalysis(BaseModel):
    area_sqft: float
    predicted_price: float
    market_value_assessment: str
    comparable_properties: List[Dict[str, float]]
    investment_recommendation: str

class PricePredictionModel:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.feature_importance = {}
        
        # Model isimleri
        self.model_names = ['xgboost', 'random_forest', 'gradient_boosting', 'linear_regression']
        
    def generate_sample_data(self, n_samples=5000):
        """Örnek emlak verisi oluştur"""
        np.random.seed(42)
        
        # Şehir ve ilçe verileri
        cities = ['Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Antalya']
        districts = {
            'Istanbul': ['Besiktas', 'Kadikoy', 'Sisli', 'Beyoglu', 'Uskudar', 'Bakirkoy'],
            'Ankara': ['Cankaya', 'Kecioren', 'Yenimahalle', 'Mamak', 'Etimesgut'],
            'Izmir': ['Konak', 'Bornova', 'Karsiyaka', 'Buca', 'Gaziemir'],
            'Bursa': ['Nilufer', 'Osmangazi', 'Yildirim', 'Gursu'],
            'Antalya': ['Muratpasa', 'Kepez', 'Konyaalti', 'Aksu']
        }
        
        # Temel özellikler
        data = []
        
        for _ in range(n_samples):
            # Şehir ve ilçe seç
            city = np.random.choice(cities)
            district = np.random.choice(districts[city])
            
            # Temel özellikler
            area_sqft = np.random.normal(1200, 400)
            area_sqft = max(400, min(area_sqft, 5000))  # 400-5000 sqft arası
            
            bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.15, 0.05])
            bathrooms = bedrooms * 0.5 + np.random.choice([0, 0.5, 1])
            bathrooms = max(1, min(bathrooms, bedrooms + 1))
            
            floors = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
            age_years = np.random.exponential(15)
            age_years = max(0, min(age_years, 50))
            
            # Lokasyon özellikleri
            neighborhood_score = np.random.normal(6, 2)
            neighborhood_score = max(1, min(neighborhood_score, 10))
            
            # Yapısal özellikler
            property_type = np.random.choice(['apartment', 'house', 'villa', 'penthouse'], 
                                           p=[0.6, 0.25, 0.1, 0.05])
            construction_quality = np.random.choice(['low', 'medium', 'high', 'luxury'], 
                                                  p=[0.2, 0.4, 0.3, 0.1])
            heating_type = np.random.choice(['central', 'individual', 'floor', 'none'], 
                                          p=[0.4, 0.3, 0.25, 0.05])
            parking_spaces = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.25, 0.05])
            
            # Mesafeler (km)
            distance_to_center = np.random.exponential(15)
            distance_to_center = max(1, min(distance_to_center, 50))
            
            distance_to_metro = np.random.exponential(2)
            distance_to_metro = max(0.1, min(distance_to_metro, 20))
            
            distance_to_school = np.random.exponential(1)
            distance_to_school = max(0.1, min(distance_to_school, 10))
            
            distance_to_hospital = np.random.exponential(3)
            distance_to_hospital = max(0.5, min(distance_to_hospital, 15))
            
            distance_to_mall = np.random.exponential(5)
            distance_to_mall = max(0.5, min(distance_to_mall, 25))
            
            # Boolean özellikler
            has_garden = np.random.choice([True, False], p=[0.3, 0.7])
            has_balcony = np.random.choice([True, False], p=[0.7, 0.3])
            has_elevator = np.random.choice([True, False], p=[0.6, 0.4])
            has_security = np.random.choice([True, False], p=[0.4, 0.6])
            has_gym = np.random.choice([True, False], p=[0.2, 0.8])
            has_pool = np.random.choice([True, False], p=[0.15, 0.85])
            
            # Pazar özellikleri
            market_trend = np.random.choice(['rising', 'stable', 'declining'], p=[0.4, 0.4, 0.2])
            season = np.random.choice(['spring', 'summer', 'autumn', 'winter'])
            
            # Fiyat hesaplama (gerçekçi faktörler)
            base_price = 0
            
            # Alan etkisi
            base_price += area_sqft * 150  # Base price per sqft
            
            # Şehir etkisi
            city_multipliers = {'Istanbul': 2.0, 'Ankara': 1.3, 'Izmir': 1.4, 'Bursa': 1.1, 'Antalya': 1.2}
            base_price *= city_multipliers[city]
            
            # Mahalle skoru etkisi
            base_price *= (0.7 + neighborhood_score * 0.05)
            
            # Yaş etkisi (yeni evler daha pahalı)
            age_factor = max(0.6, 1 - age_years * 0.01)
            base_price *= age_factor
            
            # Oda sayısı etkisi
            base_price *= (0.8 + bedrooms * 0.1)
            
            # Yapı kalitesi etkisi
            quality_multipliers = {'low': 0.8, 'medium': 1.0, 'high': 1.3, 'luxury': 1.8}
            base_price *= quality_multipliers[construction_quality]
            
            # Emlak tipi etkisi
            type_multipliers = {'apartment': 1.0, 'house': 1.2, 'villa': 1.8, 'penthouse': 2.2}
            base_price *= type_multipliers[property_type]
            
            # Mesafe etkileri (yakın olması fiyatı artırır)
            base_price *= (1.2 - distance_to_center * 0.01)  # Merkeze yakınlık
            base_price *= (1.1 - distance_to_metro * 0.02)   # Metroya yakınlık
            
            # Ek özellik etkileri
            if has_garden: base_price *= 1.1
            if has_balcony: base_price *= 1.05
            if has_elevator: base_price *= 1.08
            if has_security: base_price *= 1.06
            if has_gym: base_price *= 1.12
            if has_pool: base_price *= 1.15
            if parking_spaces > 0: base_price *= (1 + parking_spaces * 0.05)
            
            # Pazar trendi etkisi
            trend_multipliers = {'rising': 1.1, 'stable': 1.0, 'declining': 0.9}
            base_price *= trend_multipliers[market_trend]
            
            # Mevsim etkisi
            season_multipliers = {'spring': 1.05, 'summer': 1.08, 'autumn': 1.0, 'winter': 0.95}
            base_price *= season_multipliers[season]
            
            # Random noise
            noise = np.random.normal(1, 0.1)
            base_price *= max(0.7, min(noise, 1.3))
            
            # Final price
            price = max(50000, base_price)  # Minimum price
            
            data.append({
                'area_sqft': area_sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'floors': floors,
                'age_years': age_years,
                'city': city,
                'district': district,
                'neighborhood_score': neighborhood_score,
                'property_type': property_type,
                'construction_quality': construction_quality,
                'heating_type': heating_type,
                'parking_spaces': parking_spaces,
                'distance_to_center_km': distance_to_center,
                'distance_to_metro_km': distance_to_metro,
                'distance_to_school_km': distance_to_school,
                'distance_to_hospital_km': distance_to_hospital,
                'distance_to_mall_km': distance_to_mall,
                'has_garden': has_garden,
                'has_balcony': has_balcony,
                'has_elevator': has_elevator,
                'has_security': has_security,
                'has_gym': has_gym,
                'has_pool': has_pool,
                'market_trend': market_trend,
                'season': season,
                'price': price
            })
        
        return pd.DataFrame(data)
    
    def feature_engineering(self, df):
        """Feature engineering"""
        data = df.copy()
        
        # Yeni özellikler türet
        data['price_per_sqft'] = data['price'] / data['area_sqft']
        data['rooms_per_floor'] = data['bedrooms'] / data['floors']
        data['bathroom_bedroom_ratio'] = data['bathrooms'] / data['bedrooms']
        
        # Alan kategorileri
        data['size_category'] = pd.cut(data['area_sqft'], 
                                     bins=[0, 800, 1200, 2000, float('inf')], 
                                     labels=['small', 'medium', 'large', 'extra_large'])
        
        # Yaş kategorileri
        data['age_category'] = pd.cut(data['age_years'], 
                                    bins=[0, 5, 15, 30, float('inf')], 
                                    labels=['new', 'recent', 'mature', 'old'])
        
        # Lokasyon skoru (mesafelerin kombinasyonu)
        data['location_score'] = (
            (10 - np.minimum(data['distance_to_center_km'], 10)) * 0.3 +
            (5 - np.minimum(data['distance_to_metro_km'], 5)) * 0.25 +
            (3 - np.minimum(data['distance_to_school_km'], 3)) * 0.2 +
            (5 - np.minimum(data['distance_to_hospital_km'], 5)) * 0.15 +
            (10 - np.minimum(data['distance_to_mall_km'], 10)) * 0.1
        )
        
        # Lüks skoru
        luxury_features = ['has_garden', 'has_elevator', 'has_security', 'has_gym', 'has_pool']
        data['luxury_score'] = data[luxury_features].sum(axis=1)
        
        # Toplam oda sayısı
        data['total_rooms'] = data['bedrooms'] + data['bathrooms']
        
        # Kat başına alan
        data['area_per_floor'] = data['area_sqft'] / data['floors']
        
        return data
    
    def preprocess_data(self, df, fit_encoders=False):
        """Veri ön işleme"""
        data = self.feature_engineering(df)
        
        # Kategorik değişkenleri encode et
        categorical_features = ['city', 'district', 'property_type', 'construction_quality', 
                              'heating_type', 'market_trend', 'season', 'size_category', 'age_category']
        
        for feature in categorical_features:
            if feature in data.columns:
                if fit_encoders or feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    data[feature] = self.label_encoders[feature].fit_transform(data[feature].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = set(self.label_encoders[feature].classes_)
                    data[feature] = data[feature].astype(str)
                    data[feature] = data[feature].apply(
                        lambda x: x if x in unique_values else self.label_encoders[feature].classes_[0]
                    )
                    data[feature] = self.label_encoders[feature].transform(data[feature])
        
        # Boolean değişkenleri int'e çevir
        boolean_features = ['has_garden', 'has_balcony', 'has_elevator', 'has_security', 'has_gym', 'has_pool']
        for feature in boolean_features:
            if feature in data.columns:
                data[feature] = data[feature].astype(int)
        
        return data
    
    def train_models(self, df):
        """Birden fazla model eğit"""
        # Veriyi işle
        data = self.preprocess_data(df, fit_encoders=True)
        
        # Feature ve target ayır
        target_col = 'price'
        feature_cols = [col for col in data.columns if col not in [target_col, 'price_per_sqft']]
        
        X = data[feature_cols]
        y = data[target_col]
        
        self.feature_columns = feature_cols
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model tanımları
        models_config = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': Ridge(alpha=1.0)
        }
        
        # Modelleri eğit ve değerlendir
        model_scores = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            if name == 'linear_regression':
                # Linear model için scaled data kullan
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # Tree-based modeller için original data kullan
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Performans metrikleri
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: {np.sqrt(mse):.2f}")
            print(f"MAE: {mae:.2f}")
            
            # Modeli kaydet
            self.models[name] = model
            
            # Feature importance (tree-based modeller için)
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_cols, model.feature_importances_))
                self.feature_importance[name] = importance
        
        # En iyi modeli seç
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
        print(f"\nBest model: {best_model_name} (R² = {model_scores[best_model_name]['r2']:.4f})")
        
        # Ensemble model oluştur (basit ortalama)
        self.create_ensemble_model(X_test, y_test)
        
        return model_scores
    
    def create_ensemble_model(self, X_test, y_test):
        """Ensemble model oluştur"""
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'linear_regression':
                X_test_scaled = self.scaler.transform(X_test)
                pred = model.predict(X_test_scaled)
            else:
                pred = model.predict(X_test)
            predictions[name] = pred
        
        # Weighted average (R² skorlarına göre ağırlık)
        weights = {}
        total_r2 = 0
        
        for name in predictions.keys():
            r2 = r2_score(y_test, predictions[name])
            weights[name] = max(0, r2)  # Negatif R² skorları 0 yap
            total_r2 += weights[name]
        
        # Normalize weights
        if total_r2 > 0:
            for name in weights.keys():
                weights[name] /= total_r2
        else:
            # Eşit ağırlık ver
            for name in weights.keys():
                weights[name] = 1.0 / len(weights)
        
        # Ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Ensemble performansı
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        print(f"\nEnsemble Model Performance:")
        print(f"R² Score: {ensemble_r2:.4f}")
        print(f"RMSE: {ensemble_rmse:.2f}")
        
        self.ensemble_weights = weights
        
    def predict_price(self, property_data):
        """Fiyat tahmini yap"""
        if not self.models:
            raise ValueError("Models not trained yet")
        
        # Veriyi işle
        df_single = pd.DataFrame([property_data])
        df_processed = self.preprocess_data(df_single)
        
        X = df_processed[self.feature_columns]
        
        # Tüm modellerden tahmin al
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'linear_regression':
                X_scaled = self.scaler.transform(X)
                pred = model.predict(X_scaled)[0]
            else:
                pred = model.predict(X)[0]
            predictions[name] = pred
        
        # Ensemble prediction
        ensemble_pred = 0
        for name, pred in predictions.items():
            ensemble_pred += self.ensemble_weights[name] * pred
        
        # Confidence interval (basit yaklaşım)
        pred_std = np.std(list(predictions.values()))
        confidence_interval = (
            ensemble_pred - 1.96 * pred_std,
            ensemble_pred + 1.96 * pred_std
        )
        
        # Model confidence (predictions arasındaki tutarlılık)
        model_confidence = 1.0 - (pred_std / ensemble_pred) if ensemble_pred > 0 else 0.5
        model_confidence = max(0.1, min(model_confidence, 1.0))
        
        return {
            'predicted_price': ensemble_pred,
            'confidence_interval': confidence_interval,
            'model_confidence': model_confidence,
            'individual_predictions': predictions
        }
    
    def get_feature_importance(self, top_k=10):
        """En önemli özellikleri getir"""
        if not self.feature_importance:
            return {}
        
        # Ortalama feature importance hesapla
        avg_importance = {}
        for feature in self.feature_columns:
            importance_sum = 0
            count = 0
            for model_name, importance_dict in self.feature_importance.items():
                if feature in importance_dict:
                    importance_sum += importance_dict[feature]
                    count += 1
            if count > 0:
                avg_importance[feature] = importance_sum / count
        
        # Top K özellik
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_k])
    
    def save_models(self):
        """Modelleri kaydet"""
        models_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'ensemble_weights': getattr(self, 'ensemble_weights', {})
        }
        
        joblib.dump(models_data, MODELS_DIR / 'price_prediction_models.pkl')
        print("Models saved successfully!")
    
    def load_models(self):
        """Modelleri yükle"""
        try:
            models_data = joblib.load(MODELS_DIR / 'price_prediction_models.pkl')
            self.models = models_data['models']
            self.scaler = models_data['scaler']
            self.label_encoders = models_data['label_encoders']
            self.feature_columns = models_data['feature_columns']
            self.feature_importance = models_data['feature_importance']
            self.ensemble_weights = models_data.get('ensemble_weights', {})
            print("Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved models found.")
            return False

# Global model instance
model = PricePredictionModel()

# FastAPI app
app = FastAPI(
    title="Fiyat Tahmini ve Regresyon Analizi API",
    description="Emlak fiyat tahmini için ML servisi",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında modelleri yükle veya eğit"""
    if not model.load_models():
        print("Training new models...")
        
        # Örnek veri oluştur
        df = model.generate_sample_data(5000)
        df.to_csv(DATA_DIR / 'property_data.csv', index=False)
        
        # Modelleri eğit
        model.train_models(df)
        
        # Modelleri kaydet
        model.save_models()
        
        print("Models trained and saved!")

@app.get("/")
async def root():
    return {
        "message": "Fiyat Tahmini ve Regresyon Analizi API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_price": "POST /predict/price",
            "analyze_market": "POST /analyze/market",
            "feature_importance": "GET /features/importance",
            "retrain": "POST /retrain",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": len(model.models) > 0,
        "available_models": list(model.models.keys())
    }

@app.post("/predict/price", response_model=PricePrediction)
async def predict_price(property_input: PropertyInput):
    """Emlak fiyat tahmini"""
    try:
        property_dict = property_input.dict()
        result = model.predict_price(property_dict)
        
        # Price per sqft hesapla
        price_per_sqft = result['predicted_price'] / property_input.area_sqft
        
        # Market analysis
        market_analysis = {
            "value_assessment": "Fair" if 100 <= price_per_sqft <= 300 else "High" if price_per_sqft > 300 else "Low",
            "investment_potential": "Good" if property_input.neighborhood_score > 7 else "Average",
            "location_rating": "Excellent" if property_input.distance_to_center_km < 10 else "Good"
        }
        
        # Feature importance
        top_features = model.get_feature_importance(5)
        
        return PricePrediction(
            predicted_price=round(result['predicted_price'], 2),
            confidence_interval=(
                round(result['confidence_interval'][0], 2),
                round(result['confidence_interval'][1], 2)
            ),
            model_confidence=round(result['model_confidence'], 4),
            price_per_sqft=round(price_per_sqft, 2),
            market_analysis=market_analysis,
            feature_importance={k: round(v, 4) for k, v in top_features.items()}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/market", response_model=MarketAnalysis)
async def analyze_market(property_input: PropertyInput):
    """Pazar analizi"""
    try:
        property_dict = property_input.dict()
        result = model.predict_price(property_dict)
        
        predicted_price = result['predicted_price']
        
        # Market value assessment
        price_per_sqft = predicted_price / property_input.area_sqft
        
        if price_per_sqft < 150:
            value_assessment = "Undervalued - Good investment opportunity"
        elif price_per_sqft > 250:
            value_assessment = "Overvalued - Consider negotiation"
        else:
            value_assessment = "Fairly priced - Market value"
        
        # Comparable properties (simulated)
        comparable_properties = []
        base_area = property_input.area_sqft
        
        for i in range(3):
            comp_area = base_area * (0.9 + i * 0.1)
            comp_price = predicted_price * (comp_area / base_area) * (0.95 + i * 0.05)
            comparable_properties.append({
                "area_sqft": round(comp_area, 0),
                "estimated_price": round(comp_price, 2),
                "price_per_sqft": round(comp_price / comp_area, 2)
            })
        
        # Investment recommendation
        factors = []
        if property_input.neighborhood_score > 7:
            factors.append("High neighborhood score")
        if property_input.distance_to_metro_km < 2:
            factors.append("Close to metro")
        if property_input.age_years < 10:
            factors.append("Relatively new property")
        if property_input.market_trend == "rising":
            factors.append("Rising market trend")
        
        if len(factors) >= 3:
            investment_recommendation = "Strong Buy - Multiple positive factors"
        elif len(factors) >= 2:
            investment_recommendation = "Buy - Good investment potential"
        elif len(factors) >= 1:
            investment_recommendation = "Hold - Average investment"
        else:
            investment_recommendation = "Caution - Consider other options"
        
        return MarketAnalysis(
            area_sqft=property_input.area_sqft,
            predicted_price=round(predicted_price, 2),
            market_value_assessment=value_assessment,
            comparable_properties=comparable_properties,
            investment_recommendation=investment_recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/features/importance")
async def get_feature_importance(top_k: int = 15):
    """Feature importance analizi"""
    try:
        importance = model.get_feature_importance(top_k)
        
        return {
            "top_features": importance,
            "total_features": len(model.feature_columns),
            "model_types": list(model.models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
async def retrain_models():
    """Modelleri yeniden eğit"""
    try:
        # Yeni veri oluştur
        df = model.generate_sample_data(6000)
        df.to_csv(DATA_DIR / 'property_data_new.csv', index=False)
        
        # Modelleri yeniden eğit
        scores = model.train_models(df)
        
        # Modelleri kaydet
        model.save_models()
        
        return {
            "status": "success", 
            "message": "Models retrained successfully",
            "model_scores": scores
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)

