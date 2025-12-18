def classify_fraud_type(engineered_features: dict) -> str:

    # Determina el tipo de fraude basado en patrones comunes
    
    if engineered_features.get("distance_from_home", 0) > 500:
        return "Ubicación inusual"

    if engineered_features.get("amt_ratio", 0) > 3:
        return "Monto atípico"

    if engineered_features.get("geographic_velocity", 0) > 1000:
        return "Movimiento geográfico imposible"

    if engineered_features.get("merchant_risk_score", 0) > 0.8:
        return "Comercio de alto riesgo"

    if engineered_features.get("is_night", 0) == 1:
        return "Horario sospechoso"

    return "Patrón fraudulento general"


def generate_explanation(engineered_features: dict) -> dict:

    # Genera explicación clara y razones para demo

    reasons = []

    if engineered_features.get("amt_ratio", 0) > 3:
        reasons.append("El monto es significativamente mayor al promedio del usuario")

    if engineered_features.get("distance_from_home", 0) > 500:
        reasons.append("La transacción ocurrió lejos de la ubicación habitual")

    if engineered_features.get("geographic_velocity", 0) > 1000:
        reasons.append("La velocidad de desplazamiento entre transacciones es irreal")

    if engineered_features.get("merchant_risk_score", 0) > 0.8:
        reasons.append("El comercio tiene un historial alto de fraude")

    if engineered_features.get("is_night", 0) == 1:
        reasons.append("La transacción ocurrió en horario nocturno")

    if not reasons:
        reasons.append("Se detectó un patrón anómalo en el comportamiento")

    return {
        "tipo_fraude": classify_fraud_type(engineered_features),
        "explicacion": "La transacción presenta características comunes en fraudes financieros",
        "razones": reasons
    }
