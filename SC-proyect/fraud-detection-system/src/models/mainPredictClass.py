import pickle
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class FraudPredictor:
    """Predictor de fraude para transacciones individuales"""
    
    def __init__(self, model_path: str = 'models/fraud_model.pkl'):
        print(f"Cargando modelo desde: {model_path}")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.pipeline = data['pipeline']
        self.optimal_threshold = data['optimal_threshold']
        self.feature_columns = data.get('feature_columns', None)
        
        self.feature_importance = data.get('feature_importance', None)
        
        print(f"✓ Modelo cargado exitosamente")
        print(f"  Threshold óptimo: {self.optimal_threshold:.4f}")
        
        if self.feature_columns:
            print(f"  Features requeridas: {len(self.feature_columns)}")
        else:
            print(" Advertencia: Modelo sin información de columnas. Re-entrena el modelo.")
    
    def create_transaction_dataframe(self, features: Dict) -> pd.DataFrame:
        """
        Crea un DataFrame con las features de una transacción.
        Las columnas deben coincidir EXACTAMENTE con las del modelo entrenado.
        
        Args:
            features: Diccionario con las features de la transacción
            
        Returns:
            DataFrame con las columnas del modelo entrenado
        """
        if self.feature_columns is None:
            raise ValueError("El modelo no tiene información de columnas. Re-entrena el modelo.")
        
        df = pd.DataFrame(0.0, index=[0], columns=self.feature_columns, dtype='float64')
        
        for key, value in features.items():
            if key in df.columns:
                if isinstance(value, bool):
                    value = int(value)
                df.at[0, key] = value
            else:
                print(f"Feature '{key}' no existe en el modelo entrenado, se ignora")
        
        return df
    
    def predict_fraud(self, features: Dict) -> Tuple[bool, float, Dict]:
        """
        Predice si una transacción es fraudulenta.
        
        Args:
            features: Diccionario con las features de la transacción
            
        Returns:
            Tupla con (es_fraude, probabilidad, detalles)
        """
        
        df = self.create_transaction_dataframe(features)
        
        probability = self.pipeline.predict_proba(df)[:, 1][0]
        
        is_fraud = probability >= self.optimal_threshold
        
        details = {
            'is_fraud': bool(is_fraud),
            'probability': float(probability),
            'threshold': float(self.optimal_threshold),
            'confidence': float(abs(probability - 0.5) * 2),  # 0 = incierto, 1 = muy seguro
            'risk_level': self._get_risk_level(probability)
        }
        
        return is_fraud, probability, details
    
    def _get_risk_level(self, probability: float) -> str:
        """Clasifica el nivel de riesgo basado en la probabilidad"""
        if probability < 0.2:
            return "MUY BAJO"
        elif probability < 0.4:
            return "BAJO"
        elif probability < 0.6:
            return "MEDIO"
        elif probability < 0.8:
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
    
    def explain_prediction(self, features: Dict) -> str:
        """
        Genera una explicación legible de la predicción.
        
        Args:
            features: Diccionario con las features de la transacción
            
        Returns:
            String con la explicación
        """
        is_fraud, probability, details = self.predict_fraud(features)
        
        explanation = f"""
{'='*60}
ANÁLISIS DE TRANSACCIÓN
{'='*60}

RESULTADO: {'FRAUDE DETECTADO' if is_fraud else '✓ TRANSACCIÓN LEGÍTIMA'}

Probabilidad de fraude: {probability:.2%}
Nivel de riesgo: {details['risk_level']}
Confianza del modelo: {details['confidence']:.2%}
Threshold de decisión: {details['threshold']:.2%}

{'='*60}
FEATURES DE LA TRANSACCIÓN
{'='*60}
"""
        
        categories = {
            'Temporales': ['hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 
                          'time_since_last_trans', 'trans_velocity'],
            'Geográficas': ['distance_from_home', 'distance_from_last', 
                           'geographic_velocity', 'location_entropy'],
            'Comportamiento': ['amt_mean_7d', 'amt_std_7d', 'trans_freq_7d', 
                              'merchant_diversity', 'category_entropy'],
            'NLP': ['merchant_risk_score', 'has_suspicious_keyword']
        }
        
        for category, feature_list in categories.items():
            category_features = {f: features[f] for f in feature_list if f in features}
            if category_features:
                explanation += f"\n{category}:\n"
                for feature, value in category_features.items():
                    explanation += f"  • {feature}: {value}\n"
        
        explanation += f"\n{'='*60}\n"
        
        return explanation