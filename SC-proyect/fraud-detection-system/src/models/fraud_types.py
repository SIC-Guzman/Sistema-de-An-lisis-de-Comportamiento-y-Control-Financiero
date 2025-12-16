"""
Definicion de types
Aqui lo que hacemos en definir los typos de fraudes que vamos a manejar

Este es el primer archivo del conjunto de clasificacion de fraudes que consisten en 
- fraud_types.py (este)
- fraud_rules (define reglas puras)
- fraud_type_classifier (Desición)
- mainPredictClass (no tocamos modelo ni probabilidades, añadimos la clasificacion de los fraudes)
"""

from enum import Enum


class FraudType(str, Enum):
    # Fraudes soportados:

    GASTO_ATIPICO = "GASTO_ATIPICO"
    UBICACION_INUSUAL = "UBICACION_INUSUAL"
    HORARIO_INUSUAL = "HORARIO_INUSUAL"
    ACTIVIDAD_REPETITIVA = "ACTIVIDAD_REPETITIVA"

    # Caso fallback cuando no se puede clasificar
    NO_CLASIFICADO = "NO_CLASIFICADO"


# Prioridad de evaluación (orden importa)
FRAUD_TYPE_PRIORITY = [
    FraudType.GASTO_ATIPICO,
    FraudType.UBICACION_INUSUAL,
    FraudType.HORARIO_INUSUAL,
    FraudType.ACTIVIDAD_REPETITIVA,
]
