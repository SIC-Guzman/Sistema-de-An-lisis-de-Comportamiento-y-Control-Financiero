import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mainPredictClass import FraudPredictor

# Cargar modelo
predictor = FraudPredictor('models/xgboost_fraud_model.pkl')

print("\n" + "="*70)
print("SISTEMA DE DETECCION DE FRAUDE")
print(f"Precision del modelo: 89.08% | Recall: 67.05%")
print("="*70)

def ejecutar_prueba(nombre_prueba, transaccion, descripcion=None):
    """Ejecuta una prueba de prediccion"""
    print("\n" + "="*70)
    print(f"TEST: {nombre_prueba}")
    print("="*70)
    
    if descripcion:
        print(f"\n{descripcion}\n")
    
    try:
        is_fraud, prob, det = predictor.predict_fraud(transaccion)
        
        print(f"{'FRAUDE DETECTADO' if is_fraud else 'TRANSACCION LEGITIMA'}")
        print(f"\nProbabilidad: {prob:.4f} ({prob:.2%})")
        print(f"Nivel de riesgo: {det['risk_level']}")
        print(f"Threshold: {det['threshold']:.4f}")
        
        # Mostrar features clave
        eng = det['engineered_features']
        print(f"\nFactores clave:")
        print(f"  - Distancia: {eng.get('distance_from_home', 0):.2f} km")
        print(f"  - Ratio monto: {eng.get('amt_ratio', 0):.2f}x del promedio")
        print(f"  - Risk score: {eng.get('merchant_risk_score', 0):.3f}")
        print(f"  - Velocidad geo: {eng.get('geographic_velocity', 0):.1f} km/h")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    
    return True

# ============================================================================
# CASOS DE PRUEBA
# ============================================================================

# 1. FRAUDE EXTREMO - $2500, 4000km de distancia, ciudad pequeña
print("\n" + "CASOS DE ALTO RIESGO (FRAUDES ESPERADOS)")
ejecutar_prueba(
    "FRAUDE EXTREMO",
    {
        'amt': 2500.00,
        'city_pop': 3000,
        'lat': 40.7128,
        'long': -74.0060,
        'merch_lat': 34.0522,
        'merch_long': -118.2437,
        'unix_time': int(time.time()),
        'zip': 10001
    },
    "Compra de $2,500 a 4,000 km de distancia en ciudad de 3,000 habitantes"
)

# 2. FRAUDE CLASICO - $1800 en ciudad de 500 habitantes
ejecutar_prueba(
    "FRAUDE CLASICO",
    {
        'amt': 1800.00,
        'city_pop': 500,
        'lat': 41.8781,
        'long': -87.6298,
        'merch_lat': 25.7617,
        'merch_long': -80.1918,
        'unix_time': int(time.time()) - 3600,
        'zip': 60601
    },
    "Compra de $1,800 en pueblo de 500 habitantes, a 1,900 km"
)

# 3. COMPRA NOCTURNA GRANDE Y LEJANA - $950, 3am, 500km
ejecutar_prueba(
    "COMPRA NOCTURNA SOSPECHOSA",
    {
        'amt': 950.00,
        'city_pop': 25000,
        'lat': 37.7749,
        'long': -122.4194,
        'merch_lat': 32.7157,
        'merch_long': -117.1611,
        'unix_time': int(time.time()) - 82800,  # 3am hace 23 horas
        'zip': 94102
    },
    "Compra de $950 a las 3am, a 500 km de distancia"
)

# 4. MONTO EXTREMO - $5000 en ubicacion improbable
ejecutar_prueba(
    "MONTO EXTREMO",
    {
        'amt': 5000.00,
        'city_pop': 1000,
        'lat': 44.9778,
        'long': -93.2650,
        'merch_lat': 29.7604,
        'merch_long': -95.3698,
        'unix_time': int(time.time()) - 1800,
        'zip': 55401
    },
    "Compra de $5,000 en pueblo de 1,000 habitantes, hace 30 minutos"
)

# ============================================================================
print("\n" + "CASOS AMBIGUOS (SOSPECHOSOS)")
# ============================================================================

# 5. COMPRA MEDIA-ALTA A DISTANCIA MODERADA
ejecutar_prueba(
    "CASO SOSPECHOSO",
    {
        'amt': 650.00,
        'city_pop': 50000,
        'lat': 37.7749,
        'long': -122.4194,
        'merch_lat': 37.3382,
        'merch_long': -121.8863,
        'unix_time': int(time.time()) - 7200,
        'zip': 94102
    },
    "Compra de $650 a 70 km de distancia"
)

# 6. VIAJE CON COMPRA MODERADA
ejecutar_prueba(
    "POSIBLE VIAJE",
    {
        'amt': 285.00,
        'city_pop': 100000,
        'lat': 47.6062,
        'long': -122.3321,
        'merch_lat': 45.5152,
        'merch_long': -122.6784,
        'unix_time': int(time.time()) - 14400,
        'zip': 98101
    },
    "Compra de $285 a 280 km (Seattle-Portland), podria ser viaje legitimo"
)

# ============================================================================
print("\n" + "CASOS DE BAJO RIESGO (LEGITIMOS ESPERADOS)")
# ============================================================================

# 7. TRANSACCION NORMAL - $35 cerca de casa
ejecutar_prueba(
    "COMPRA LOCAL PEQUEÑA",
    {
        'amt': 35.50,
        'city_pop': 250000,
        'lat': 42.3601,
        'long': -71.0589,
        'merch_lat': 42.3605,
        'merch_long': -71.0595,
        'unix_time': int(time.time()) - 86400,
        'zip': 2108
    },
    "Compra de $35.50 a menos de 100 metros"
)

# 8. COMPRA TIPICA - $67 en mismo vecindario
ejecutar_prueba(
    "COMPRA RUTINARIA",
    {
        'amt': 67.80,
        'city_pop': 500000,
        'lat': 39.7392,
        'long': -104.9903,
        'merch_lat': 39.7420,
        'merch_long': -104.9920,
        'unix_time': int(time.time()) - 43200,
        'zip': 80202
    },
    "Compra de $67.80 a 300 metros en ciudad grande"
)

# 9. CIUDAD GRANDE, COMPRA PEQUEÑA
ejecutar_prueba(
    "COMPRA SEGURA",
    {
        'amt': 22.50,
        'city_pop': 1000000,
        'lat': 33.4484,
        'long': -112.0740,
        'merch_lat': 33.4490,
        'merch_long': -112.0745,
        'unix_time': int(time.time()) - 3600,
        'zip': 85001
    },
    "Compra de $22.50 en ciudad de 1M habitantes, a 100 metros"
)

# 10. MISMA UBICACION - Compra online/tienda
ejecutar_prueba(
    "COMPRA EN TIENDA",
    {
        'amt': 125.00,
        'city_pop': 350000,
        'lat': 30.2672,
        'long': -97.7431,
        'merch_lat': 30.2672,
        'merch_long': -97.7431,
        'unix_time': int(time.time()) - 600,
        'zip': 78701
    },
    "Compra de $125 en exactamente la misma ubicacion (online/presencial)"
)

# ============================================================================
print("\n" + "="*70)
print("RESUMEN DE PRUEBAS")
print("="*70)
print("El modelo ahora usa features engineered correctamente")
print("Deteccion basada en: monto, distancia, poblacion, horario, patrones")
print(f"Threshold: {predictor.optimal_threshold:.4f}")
print(f"Metricas: Precision=89.08%, Recall=67.05%")
print("="*70)