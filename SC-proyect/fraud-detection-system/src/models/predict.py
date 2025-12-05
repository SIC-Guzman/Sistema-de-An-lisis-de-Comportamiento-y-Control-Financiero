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
        
        df = pd.DataFrame(0, index=[0], columns=self.feature_columns)
        
        for key, value in features.items():
            if key in df.columns:
                # Convertir booleanos a int
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
        # Crear DataFrame con las columnas correctas
        df = self.create_transaction_dataframe(features)
        
        # Obtener probabilidad
        probability = self.pipeline.predict_proba(df)[:, 1][0]
        
        # Determinar si es fraude según el threshold
        is_fraud = probability >= self.optimal_threshold
        
        # Crear detalles adicionales
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
            # Si no hay feature importance, calcularla del modelo
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
        
        # Agrupar features por categoría
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


def ejemplo_uso():
    """Ejemplo de cómo usar el predictor"""
    
    # Inicializar predictor
    predictor = FraudPredictor('models/fraud_model.pkl')
    
    print("\n" + "="*60)
    print("EJEMPLO 1: Transacción Sospechosa")
    print("="*60)
    
    # Transacción sospechosa (noche, lejos de casa, alta velocidad)
    transaccion_sospechosa = {
        'hour': 3,  # 3 AM
        'is_night': True,
        'is_weekend': True,
        'distance_from_home': 500.0,  # 500 km de casa
        'distance_from_last': 450.0,
        'geographic_velocity': 200.0,  # 200 km/h (imposible normalmente)
        'time_since_last_trans': 7200,  # 2 horas desde última
        'merchant_risk_score': 0.8,  # Comerciante de alto riesgo
        'has_suspicious_keyword': True,
        'amt_mean_7d': 50.0,
        'amt_std_7d': 15.0,
        'location_entropy': 0.5  # Baja entropía = pocas ubicaciones
    }
    
    print(predictor.explain_prediction(transaccion_sospechosa))
    
    print("\n" + "="*60)
    print("EJEMPLO 2: Transacción Normal")
    print("="*60)
    
    # Transacción normal
    transaccion_normal = {
        'hour': 14,  # 2 PM
        'is_night': False,
        'is_weekend': False,
        'distance_from_home': 5.0,  # 5 km de casa
        'distance_from_last': 3.0,
        'geographic_velocity': 15.0,  # Velocidad normal
        'time_since_last_trans': 86400,  # 1 día desde última
        'merchant_risk_score': 0.2,  # Comerciante confiable
        'has_suspicious_keyword': False,
        'amt_mean_7d': 45.0,
        'amt_std_7d': 12.0,
        'location_entropy': 2.5  # Alta entropía = muchas ubicaciones
    }
    
    print(predictor.explain_prediction(transaccion_normal))
    
    # Mostrar features más importantes
    print("\n" + "="*60)
    print("TOP 10 FEATURES MÁS IMPORTANTES DEL MODELO")
    print("="*60)
    print(predictor.get_top_features(10).to_string(index=False))


def predecir_transaccion_custom():
    """Función interactiva para predecir transacciones personalizadas"""
    
    predictor = FraudPredictor('models/fraud_model.pkl')
    
    print("\n" + "="*60)
    print("PREDICTOR DE FRAUDE - TRANSACCIÓN PERSONALIZADA")
    print("="*60)
    print("\nIngresa los valores para la transacción:")
    print("(Presiona Enter para usar valor por defecto)\n")
    
    features = {}
    
    # Temporales
    features['hour'] = int(input("Hora del día (0-23) [12]: ") or 12)
    features['is_night'] = input("¿Es de noche (2AM-6AM)? (s/n) [n]: ").lower() == 's'
    features['is_weekend'] = input("¿Es fin de semana? (s/n) [n]: ").lower() == 's'
    
    # Geográficas
    features['distance_from_home'] = float(input("Distancia desde casa (km) [10.0]: ") or 10.0)
    features['geographic_velocity'] = float(input("Velocidad geográfica (km/h) [20.0]: ") or 20.0)
    
    # Comportamiento
    features['amt_mean_7d'] = float(input("Monto promedio últimos 7 días [50.0]: ") or 50.0)
    features['trans_freq_7d'] = int(input("Frecuencia de transacciones 7d [10]: ") or 10)
    
    # NLP
    features['merchant_risk_score'] = float(input("Score de riesgo del comerciante (0-1) [0.3]: ") or 0.3)
    features['has_suspicious_keyword'] = input("¿Tiene palabras sospechosas? (s/n) [n]: ").lower() == 's'
    
    print(predictor.explain_prediction(features))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        predecir_transaccion_custom()
    else:
        ejemplo_uso()