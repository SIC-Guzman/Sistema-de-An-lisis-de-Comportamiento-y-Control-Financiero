from flask import Flask, jsonify, render_template
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from src.models.mainPredictClass import FraudPredictor
from src.data.pipeline import DataPipeline
from src.utils.constants import Columns

# =========================
# PATHS BASE
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.pkl"
TEMPLATES_DIR = BASE_DIR / "src" / "ui" / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# =========================
# MODELO Y PIPELINE 
# =========================
predictor = FraudPredictor(str(MODEL_PATH))
pipeline = DataPipeline(BASE_DIR)

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict():
    """
    Ejecuta predicción usando el dataset en data/raw/
    """
    try:
        # 1. Cargar dataset limpio desde pipeline
        df = pipeline.load_and_prepare(dataset="test")

        predictions = []

        for _, row in df.iterrows():
            raw_features = row.to_dict()

            is_fraud, prob, details = predictor.predict_fraud(raw_features)

            unix_ts = raw_features.get(Columns.UNIX_TIME)
            fecha = None
            if pd.notna(unix_ts):
                try:
                    fecha = datetime.fromtimestamp(
                        int(unix_ts), tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    fecha = None

            predictions.append({
                "es_fraude": bool(details["is_fraud"]),
                "probabilidad_fraude": round(float(details["probability"]), 4),
                "nivel_riesgo": details["risk_level"],
                "confianza_modelo": round(float(details["confidence"]), 4),
                "fecha": fecha,
                "monto": raw_features.get(Columns.AMT),
                "tarjeta_credito": raw_features.get(Columns.CC_NUM),
            })

        # Transacciones sospechosas
        sospechosas = [p for p in predictions if p["es_fraude"]]

        response = {
            "total_transacciones": len(predictions),
            "total_sospechosas": len(sospechosas),
            "hay_alerta": len(sospechosas) > 0,
            "transacciones_sospechosas": sospechosas,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "detail": str(e)
        }), 500


if __name__ == "__main__":
    # Ejecutar desde raíz:
    # python -m src.api.app
    app.run(debug=True)
