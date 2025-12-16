import pickle
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# Nuevos importes para el Bloque Clasificacion de Fraudes
from .fraud_type_classifier import FraudTypeClassifier
from .fraud_types import FraudType


class FraudPredictor:
    """Predictor de fraude para transacciones individuales"""
    
    def __init__(self, model_path: str = 'models/xgboost_fraud_model.pkl'):
        print(f"Cargando modelo desde: {model_path}")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.pipeline = data['pipeline']
        self.optimal_threshold = data['optimal_threshold']
        self.feature_columns = data.get('feature_columns', None)

        #Adicion del Bloque Clasificacion de Fraudes
        self.fraud_type_classifier = FraudTypeClassifier()

        
        self.feature_importance = data.get('feature_importance', None)
        
        print(f"✓ Modelo cargado exitosamente")
        print(f"  Threshold óptimo: {self.optimal_threshold:.4f}")
        
        if self.feature_columns:
            print(f"  Features requeridas: {len(self.feature_columns)}")
        else:
            print("  Advertencia: Modelo sin información de columnas. Re-entrena el modelo.")
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calcula distancia en km entre dos coordenadas"""
        R = 6371  # Radio de la Tierra en km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _engineer_features(self, raw_features: Dict) -> Dict:
        """
        Convierte features raw en features engineered que el modelo espera.
        """
        features = {}
        
        # Features básicas que se pasan directamente
        if 'amt' in raw_features:
            features['amt'] = raw_features['amt']
        if 'city_pop' in raw_features:
            features['city_pop'] = raw_features['city_pop']
        if 'zip' in raw_features:
            features['zip'] = raw_features['zip']
        
        # Features temporales
        if 'unix_time' in raw_features:
            dt = datetime.fromtimestamp(raw_features['unix_time'])
            features['hour'] = dt.hour
            features['day_of_week'] = dt.weekday()
            features['month'] = dt.month
            features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            features['is_night'] = 1 if (dt.hour >= 22 or dt.hour <= 6) else 0
        
        # Features geográficas
        if all(k in raw_features for k in ['lat', 'long', 'merch_lat', 'merch_long']):
            distance = self._haversine_distance(
                raw_features['lat'], raw_features['long'],
                raw_features['merch_lat'], raw_features['merch_long']
            )
            features['distance_from_home'] = distance
            
            # Simulación de distancia desde última transacción (en producción vendría de BD)
            features['distance_from_last'] = distance * 0.5  # Aproximación
            
            # Velocidad geográfica (km/h) - asumimos 1 hora desde última transacción
            features['geographic_velocity'] = distance / 1.0
        
        # Features de comportamiento (valores simulados - en producción vendrían de BD)
        # Estos son críticos para la detección
        amt = raw_features.get('amt', 0)
        
        # Simulamos patrones basados en el monto
        if amt < 50:
            # Transacciones pequeñas - patrón normal
            features['amt_mean_7d'] = 45.0
            features['amt_std_7d'] = 15.0
            features['trans_freq_7d'] = 8.0
            features['merchant_diversity'] = 6.0
        elif amt < 200:
            # Transacciones medias
            features['amt_mean_7d'] = 120.0
            features['amt_std_7d'] = 40.0
            features['trans_freq_7d'] = 5.0
            features['merchant_diversity'] = 4.0
        elif amt < 500:
            # Transacciones grandes - más sospechoso
            features['amt_mean_7d'] = 80.0
            features['amt_std_7d'] = 50.0
            features['trans_freq_7d'] = 3.0
            features['merchant_diversity'] = 3.0
        else:
            # Transacciones muy grandes - altamente sospechoso
            features['amt_mean_7d'] = 60.0
            features['amt_std_7d'] = 30.0
            features['trans_freq_7d'] = 2.0
            features['merchant_diversity'] = 2.0
        
        # Ratio del monto actual vs promedio
        features['amt_ratio'] = amt / features['amt_mean_7d'] if features['amt_mean_7d'] > 0 else 1.0
        
        # Features adicionales
        features['time_since_last_trans'] = 3600  # 1 hora por defecto
        features['trans_velocity'] = 1.0  # 1 transacción por hora
        features['location_entropy'] = 0.5  # Entropía de ubicación
        features['category_entropy'] = 0.6  # Entropía de categoría
        
        # Features de NLP (simplificadas)
        features['merchant_risk_score'] = 0.1  # Score bajo por defecto
        features['has_suspicious_keyword'] = 0  # No sospechoso por defecto
        
        # Ajustes basados en distancia
        if 'distance_from_home' in features:
            if features['distance_from_home'] > 1000:
                # Transacción muy lejos - aumentar sospecha
                features['location_entropy'] = 0.9
                features['merchant_risk_score'] = 0.4
            elif features['distance_from_home'] < 1:
                # Transacción muy cerca - reducir sospecha
                features['location_entropy'] = 0.2
                features['merchant_risk_score'] = 0.05
        
        # Ajustes basados en población
        if 'city_pop' in features:
            if features['city_pop'] < 5000:
                # Ciudad pequeña con monto alto = sospechoso
                if amt > 500:
                    features['merchant_risk_score'] = 0.6
                    features['location_entropy'] = 0.8
        
        # Ajustes basados en horario
        if 'is_night' in features and features['is_night'] == 1:
            if amt > 300:
                # Compras grandes de noche = más sospechoso
                features['merchant_risk_score'] += 0.2
        
        return features
    
    def create_transaction_dataframe(self, features: Dict) -> pd.DataFrame:
        """
        Crea un DataFrame con las features de una transacción.
        """
        if self.feature_columns is None:
            raise ValueError("El modelo no tiene información de columnas. Re-entrena el modelo.")
        
        # Crear DataFrame con todas las columnas en 0
        df = pd.DataFrame(0.0, index=[0], columns=self.feature_columns, dtype='float64')
        
        # Llenar con features engineered
        for key, value in features.items():
            if key in df.columns:
                if isinstance(value, bool):
                    value = int(value)
                df.at[0, key] = value
        
        return df
    
    def predict_fraud(self, raw_features: Dict) -> Tuple[bool, float, Dict]:
        """
        Predice si una transacción es fraudulenta.
        
        Args:
            raw_features: Diccionario con features raw de la transacción
                         (amt, lat, long, merch_lat, merch_long, unix_time, etc.)
            
        Returns:
            Tupla con (es_fraude, probabilidad, detalles)
        """
        
        # Convertir features raw a engineered
        engineered_features = self._engineer_features(raw_features)
        
        # Crear DataFrame
        df = self.create_transaction_dataframe(engineered_features)
        
        # Predecir
        probability = self.pipeline.predict_proba(df)[:, 1][0]
        
        # Determinar is_fraud basado en rangos de probabilidad
        is_fraud = self._classify_fraud(probability)
        



        # reemplazamos para el Bloque Clasificacion de Fraudes
        # Eliminamos
        """
        details = {
            'is_fraud': bool(is_fraud),
            'probability': float(probability),
            'threshold': float(self.optimal_threshold),
            'confidence': float(abs(probability - 0.5) * 2),
            'risk_level': self._get_risk_level(probability),
            'engineered_features': engineered_features
        }
        """
        # =========================
        # Agregamos: CLASIFICACIÓN DEL TIPO DE FRAUDE 
        # =========================
        fraud_type, fraud_reasons = self.fraud_type_classifier.classify(
            is_fraud=is_fraud,
            raw_features=raw_features,
            engineered_features=engineered_features
        )

        details = {
            'is_fraud': bool(is_fraud),
            'probability': float(probability),
            'threshold': float(self.optimal_threshold),
            'confidence': float(abs(probability - 0.5) * 2),
            'risk_level': self._get_risk_level(probability),

            # Sprint 2
            'fraud_type': fraud_type.value if isinstance(fraud_type, FraudType) else str(fraud_type),
            'fraud_reasons': fraud_reasons,

            # Se mantiene para explicación / debugging
            'engineered_features': engineered_features
        }



        
        return is_fraud, probability, details
    
    def _classify_fraud(self, probability: float) -> bool:
        if probability < 0.03:
            return False  # MUY BAJO
        elif probability < 0.06:
            return False  # BAJO
        elif probability < 0.1:
            return True  # MEDIO
        elif probability < 0.:
            return True   # ALTO
        else:
            return True   # MUY ALTO
    
    def _get_risk_level(self, probability: float) -> str:
        """Clasifica el nivel de riesgo basado en la probabilidad"""
        if probability < 0.1:
            return "MUY BAJO"
        elif probability < 0.3:
            return "BAJO"
        elif probability < 0.5:
            return "MEDIO"
        elif probability < 0.7:
            return "ALTO"
        else:
            return "MUY ALTO"
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Retorna las top N features más importantes del modelo"""
        if self.feature_importance is not None:
            return self.feature_importance.head(n)
        else:
            print("Calculando feature importance del modelo...")
            model = self.pipeline.named_steps['model']
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            df = pd.DataFrame({
                'feature': [self.feature_columns[i] for i in indices[:n]],
                'importance': importances[indices[:n]]
            })
            return df
    
    def explain_prediction(self, raw_features: Dict) -> str:
        """
        Genera una explicación legible de la predicción.
        """
        is_fraud, probability, details = self.predict_fraud(raw_features)
        
        explanation = f"""
{'='*70}
ANÁLISIS DE TRANSACCIÓN
{'='*70}

RESULTADO: {'FRAUDE DETECTADO' if is_fraud else '✓ TRANSACCIÓN LEGÍTIMA'}

Probabilidad de fraude: {probability:.2%}
Nivel de riesgo: {details['risk_level']}
Confianza del modelo: {details['confidence']:.2%}
Threshold de decisión: {details['threshold']:.2%}

{'='*70}
FEATURES RAW
{'='*70}
Monto: ${raw_features.get('amt', 0):.2f}
Población ciudad: {raw_features.get('city_pop', 0):,}
"""
        
        if all(k in raw_features for k in ['lat', 'long', 'merch_lat', 'merch_long']):
            dist = self._haversine_distance(
                raw_features['lat'], raw_features['long'],
                raw_features['merch_lat'], raw_features['merch_long']
            )
            explanation += f"Distancia cliente-comerciante: {dist:.2f} km\n"
        
        if 'unix_time' in raw_features:
            dt = datetime.fromtimestamp(raw_features['unix_time'])
            explanation += f"Fecha/Hora: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        explanation += f"\n{'='*70}\n"
        explanation += "FEATURES ENGINEERED CLAVE\n"
        explanation += f"{'='*70}\n"
        
        eng_features = details['engineered_features']
        key_features = ['amt_ratio', 'distance_from_home', 'geographic_velocity', 
                       'merchant_risk_score', 'location_entropy', 'is_night']
        
        for feat in key_features:
            if feat in eng_features:
                explanation += f"  • {feat}: {eng_features[feat]:.3f}\n"
        
        explanation += f"\n{'='*70}\n"
        
        return explanation