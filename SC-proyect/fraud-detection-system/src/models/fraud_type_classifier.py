"""
Sprint 2 – Clasificador de Tipo de Fraude

Este módulo orquesta las reglas de clasificación y decide
UN SOLO tipo de fraude dominante por transacción.

No predice fraude. Solo interpreta el resultado del modelo.
"""

from typing import Dict, List, Tuple

from .fraud_types import FraudType, FRAUD_TYPE_PRIORITY
from .fraud_rules import FRAUD_RULES


class FraudTypeClassifier:
    """Clasifica el tipo de fraude basado en reglas."""

    def classify(
        self,
        is_fraud: bool,
        raw_features: Dict,
        engineered_features: Dict
    ) -> Tuple[FraudType, List[str]]:
        """
        Retorna el tipo de fraude y las razones asociadas.

        Si no es fraude, no clasifica.
        """

        if not is_fraud:
            return FraudType.NO_CLASIFICADO, []

        for fraud_type in FRAUD_TYPE_PRIORITY:
            rule_fn = FRAUD_RULES.get(fraud_type)
            if not rule_fn:
                continue

            aplica, razones = rule_fn(raw_features, engineered_features)
            if aplica:
                return fraud_type, razones

        return FraudType.NO_CLASIFICADO, ["No se pudo determinar un tipo dominante"]
