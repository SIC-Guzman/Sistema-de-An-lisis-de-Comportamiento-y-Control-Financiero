# predecir_tests.py
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Se aprovecha el archivo "predict_base" para usar su clase de predicciones ya creada
from models.mainPredictClass import FraudPredictor

# Cargar modelo una sola vez
predictor = FraudPredictor('models/fraud_model.pkl')

print("\n" + "="*70)
print("FEATURES DISPONIBLES EN EL MODELO:")
print("="*70)
print("1. Unnamed: 0    - ID de transacción")
print("2. amt           - Monto de la transacción")
print("3. cc_num        - Número de tarjeta de crédito")
print("4. city_pop      - Población de la ciudad")
print("5. lat           - Latitud del cliente")
print("6. long          - Longitud del cliente")
print("7. merch_lat     - Latitud del comerciante")
print("8. merch_long    - Longitud del comerciante")
print("9. unix_time     - Timestamp Unix")
print("10. zip          - Código postal")
print("="*70)

# ============================================================================
# ESCENARIOS DE PRUEBA CON FEATURES REALES
# ============================================================================
# Nota: el modelo aún requiere de entrenamiento para cumplir con las salidas propuestas en los ejemplos

# 1. FRAUDE EXTREMO - Monto alto + Ubicación muy lejana
print("\n" + "="*70)
print("TEST 1: FRAUDE EXTREMO - Compra Grande en Ubicación Lejana")
print("="*70)
fraude_extremo = {
    'Unnamed: 0': 1,
    'amt': 2500.00,                    # Monto muy alto
    'cc_num': 4532015112830366,
    'city_pop': 3000,                  # Ciudad pequeña
    'lat': 40.7128,                    # Nueva York (ejemplo)
    'long': -74.0060,
    'merch_lat': 34.0522,              # Los Ángeles (muy lejos)
    'merch_long': -118.2437,
    'unix_time': int(time.time()),     # Ahora
    'zip': 10001
}

es_fraude, prob, det = predictor.predict_fraud(fraude_extremo)
print(f"Monto: ${fraude_extremo['amt']:.2f}")
print(f"Distancia cliente-comerciante: ~4000 km (NY a LA)")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 2. FRAUDE CLÁSICO - Monto muy alto en ciudad pequeña
print("\n" + "="*70)
print("TEST 2: FRAUDE CLÁSICO - Compra Inusual en Población Pequeña")
print("="*70)
fraude_clasico = {
    'Unnamed: 0': 2,
    'amt': 1800.00,                    # Monto alto
    'cc_num': 4532015112830367,
    'city_pop': 500,                   # Población muy pequeña
    'lat': 41.8781,                    # Chicago
    'long': -87.6298,
    'merch_lat': 25.7617,              # Miami (lejos)
    'merch_long': -80.1918,
    'unix_time': int(time.time()) - 3600,  # Hace 1 hora
    'zip': 60601
}

es_fraude, prob, det = predictor.predict_fraud(fraude_clasico)
print(f"Monto: ${fraude_clasico['amt']:.2f}")
print(f"Población ciudad: {fraude_clasico['city_pop']}")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 3. SOSPECHOSO MODERADO - Monto medio-alto con distancia moderada
print("\n" + "="*70)
print("TEST 3: SOSPECHOSO MODERADO - Patrón Inusual")
print("="*70)
sospechoso_moderado = {
    'Unnamed: 0': 3,
    'amt': 850.00,                     # Monto medio-alto
    'cc_num': 4532015112830368,
    'city_pop': 50000,                 # Ciudad mediana
    'lat': 37.7749,                    # San Francisco
    'long': -122.4194,
    'merch_lat': 37.3382,              # San José (cerca pero no igual)
    'merch_long': -121.8863,
    'unix_time': int(time.time()) - 7200,  # Hace 2 horas
    'zip': 94102
}

es_fraude, prob, det = predictor.predict_fraud(sospechoso_moderado)
print(f"Monto: ${sospechoso_moderado['amt']:.2f}")
print(f"Población ciudad: {sospechoso_moderado['city_pop']}")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 4. TRANSACCIÓN NORMAL - Monto bajo cerca de casa
print("\n" + "="*70)
print("TEST 4: TRANSACCIÓN NORMAL - Compra Local Pequeña")
print("="*70)
normal_tipica = {
    'Unnamed: 0': 4,
    'amt': 35.50,                      # Monto bajo/normal
    'cc_num': 4532015112830369,
    'city_pop': 250000,                # Ciudad grande
    'lat': 42.3601,                    # Boston
    'long': -71.0589,
    'merch_lat': 42.3605,              # Muy cerca (misma área)
    'merch_long': -71.0595,
    'unix_time': int(time.time()) - 86400,  # Hace 24 horas
    'zip': 2108
}

es_fraude, prob, det = predictor.predict_fraud(normal_tipica)
print(f"Monto: ${normal_tipica['amt']:.2f}")
print(f"Población ciudad: {normal_tipica['city_pop']}")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 5. TRANSACCIÓN LEGÍTIMA - Compra típica en ciudad grande
print("\n" + "="*70)
print("TEST 5: TRANSACCIÓN LEGÍTIMA - Compra Rutinaria")
print("="*70)
legitima = {
    'Unnamed: 0': 5,
    'amt': 67.80,                      # Monto normal
    'cc_num': 4532015112830370,
    'city_pop': 500000,                # Ciudad grande
    'lat': 39.7392,                    # Denver
    'long': -104.9903,
    'merch_lat': 39.7420,              # Mismo vecindario
    'merch_long': -104.9920,
    'unix_time': int(time.time()) - 43200,  # Hace 12 horas
    'zip': 80202
}

es_fraude, prob, det = predictor.predict_fraud(legitima)
print(f"Monto: ${legitima['amt']:.2f}")
print(f"Población ciudad: {legitima['city_pop']}")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 6. COMPRA PEQUEÑA LEJOS - Puede ser legítima (viaje)
print("\n" + "="*70)
print("TEST 6: COMPRA PEQUEÑA EN VIAJE - Caso Ambiguo")
print("="*70)
viaje_pequeno = {
    'Unnamed: 0': 6,
    'amt': 45.00,                      # Monto pequeño
    'cc_num': 4532015112830371,
    'city_pop': 100000,
    'lat': 47.6062,                    # Seattle
    'long': -122.3321,
    'merch_lat': 45.5152,              # Portland (distancia media)
    'merch_long': -122.6784,
    'unix_time': int(time.time()) - 14400,  # Hace 4 horas
    'zip': 98101
}

es_fraude, prob, det = predictor.predict_fraud(viaje_pequeno)
print(f"Monto: ${viaje_pequeno['amt']:.2f}")
print(f"Distancia: ~280 km (viaje posible)")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 7. MONTO EXTREMO - Muy alto en ubicación improbable
print("\n" + "="*70)
print("TEST 7: MONTO EXTREMO - Compra Muy Grande")
print("="*70)
monto_extremo = {
    'Unnamed: 0': 7,
    'amt': 5000.00,                    # Monto extremadamente alto
    'cc_num': 4532015112830372,
    'city_pop': 1000,                  # Pueblo pequeño
    'lat': 44.9778,                    # Minneapolis
    'long': -93.2650,
    'merch_lat': 29.7604,              # Houston (muy lejos)
    'merch_long': -95.3698,
    'unix_time': int(time.time()) - 1800,  # Hace 30 minutos
    'zip': 55401
}

es_fraude, prob, det = predictor.predict_fraud(monto_extremo)
print(f"Monto: ${monto_extremo['amt']:.2f}")
print(f"Población: {monto_extremo['city_pop']} (pueblo pequeño)")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 8. CIUDAD GRANDE COMPRA NORMAL - Muy seguro
print("\n" + "="*70)
print("TEST 8: COMPRA SEGURA - Ciudad Grande, Monto Bajo")
print("="*70)
muy_segura = {
    'Unnamed: 0': 8,
    'amt': 22.50,                      # Monto muy bajo
    'cc_num': 4532015112830373,
    'city_pop': 1000000,               # Ciudad muy grande
    'lat': 33.4484,                    # Phoenix
    'long': -112.0740,
    'merch_lat': 33.4490,              # Prácticamente mismo lugar
    'merch_long': -112.0745,
    'unix_time': int(time.time()) - 3600,  # Hace 1 hora
    'zip': 85001
}

es_fraude, prob, det = predictor.predict_fraud(muy_segura)
print(f"Monto: ${muy_segura['amt']:.2f}")
print(f"Población ciudad: {muy_segura['city_pop']:,}")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 9. COORDENADAS IDÉNTICAS - Compra online o en tienda
print("\n" + "="*70)
print("TEST 9: MISMA UBICACIÓN - Online o Tienda Física")
print("="*70)
misma_ubicacion = {
    'Unnamed: 0': 9,
    'amt': 125.00,                     # Monto medio
    'cc_num': 4532015112830374,
    'city_pop': 350000,
    'lat': 30.2672,                    # Austin
    'long': -97.7431,
    'merch_lat': 30.2672,              # Exactamente mismo lugar
    'merch_long': -97.7431,
    'unix_time': int(time.time()) - 600,   # Hace 10 minutos
    'zip': 78701
}

es_fraude, prob, det = predictor.predict_fraud(misma_ubicacion)
print(f"Monto: ${misma_ubicacion['amt']:.2f}")
print(f"Distancia: 0 km (mismo lugar)")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

# 10. CASO LÍMITE - Monto medio en distancia media
print("\n" + "="*70)
print("TEST 10: CASO LÍMITE - Parámetros Intermedios")
print("="*70)
caso_limite = {
    'Unnamed: 0': 10,
    'amt': 450.00,                     # Monto medio-alto
    'cc_num': 4532015112830375,
    'city_pop': 75000,                 # Ciudad mediana
    'lat': 38.5816,                    # Sacramento
    'long': -121.4944,
    'merch_lat': 37.7749,              # San Francisco (distancia media)
    'merch_long': -122.4194,
    'unix_time': int(time.time()) - 10800, # Hace 3 horas
    'zip': 95814
}

es_fraude, prob, det = predictor.predict_fraud(caso_limite)
print(f"Monto: ${caso_limite['amt']:.2f}")
print(f"Población: {caso_limite['city_pop']:,}")
print(f"Distancia: ~140 km")
print(f"Resultado: {' FRAUDE' if es_fraude else '✓ Legítimo'}")
print(f"Probabilidad: {prob:.2%} | Riesgo: {det['risk_level']}")

print("\n" + "="*70)
print("PRUEBAS COMPLETADAS")
print("="*70)
print("\nNOTA: El modelo usa solo features básicas sin ingeniería de features.")