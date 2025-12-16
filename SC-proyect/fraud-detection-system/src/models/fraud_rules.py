"""
Definimos nuestras reglas

Aqui definimos que reglas tenemos para poder clasificar nuestros fraudes... como entrada tenemos:

- raw_features (dict)
- engineered_features (dict)

Y como salida:
- (bool aplica, list razones)
"""

from typing import Dict, List, Tuple

from .fraud_types import FraudType

# =========================
# UMBRALES (AJUSTABLES)
# =========================
# podemos ajustar dependiendo del modelo, sujeto a parametros..
AMT_RATIO_HIGH = 3.0
DISTANCE_HIGH_KM = 500.0
NIGHT_AMT_THRESHOLD = 300.0
TRANS_FREQ_HIGH = 8.0


# =========================
# REGLAS INDIVIDUALES
# =========================

def regla_gasto_atipico(raw: Dict, eng: Dict) -> Tuple[bool, List[str]]:
    razones = []

    amt_ratio = eng.get("amt_ratio")
    if amt_ratio is not None and amt_ratio >= AMT_RATIO_HIGH:
        razones.append(f"Monto {amt_ratio:.1f}x mayor al promedio del usuario")
        return True, razones

    return False, []


def regla_ubicacion_inusual(raw: Dict, eng: Dict) -> Tuple[bool, List[str]]:
    razones = []

    distancia = eng.get("distance_from_home")
    if distancia is not None and distancia >= DISTANCE_HIGH_KM:
        razones.append(f"Transacción a {distancia:.0f} km del lugar habitual")
        return True, razones

    return False, []


def regla_horario_inusual(raw: Dict, eng: Dict) -> Tuple[bool, List[str]]:
    razones = []

    is_night = eng.get("is_night")
    amt = raw.get("amt")

    if is_night == 1 and amt is not None and amt >= NIGHT_AMT_THRESHOLD:
        razones.append("Compra nocturna con monto elevado")
        return True, razones

    return False, []


def regla_actividad_repetitiva(raw: Dict, eng: Dict) -> Tuple[bool, List[str]]:
    razones = []

    freq = eng.get("trans_freq_7d")
    if freq is not None and freq >= TRANS_FREQ_HIGH:
        razones.append(f"Alta frecuencia de transacciones recientes ({freq})")
        return True, razones

    return False, []


# =========================
# MAPA DE REGLAS
# =========================
FRAUD_RULES = {
    FraudType.GASTO_ATIPICO: regla_gasto_atipico,
    FraudType.UBICACION_INUSUAL: regla_ubicacion_inusual,
    FraudType.HORARIO_INUSUAL: regla_horario_inusual,
    FraudType.ACTIVIDAD_REPETITIVA: regla_actividad_repetitiva,
}
