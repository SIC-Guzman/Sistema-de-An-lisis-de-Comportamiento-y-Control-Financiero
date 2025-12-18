from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.utils.explanations import generate_explanation
from pathlib import Path
from datetime import datetime, timezone
import io

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
CORS(app)  # Habilitar CORS para peticiones desde el frontend

# =========================
# MODELO Y PIPELINE 
# =========================
predictor = FraudPredictor(str(MODEL_PATH))
pipeline = DataPipeline(BASE_DIR)

# =========================
# MAPEO DE COLUMNAS (español -> inglés para el modelo)
# =========================
COLUMN_MAPPING_ES_TO_EN = {
    'id_transaccion': 'Unnamed: 0',
    'monto': 'amt',
    'tarjeta_credito': 'cc_num',
    'poblacion_ciudad': 'city_pop',
    'latitud_cliente': 'lat',
    'longitud_cliente': 'long',
    'latitud_comercio': 'merch_lat',
    'longitud_comercio': 'merch_long',
    'tiempo_unix': 'unix_time',
    'codigo_postal': 'zip'
}

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Ejecuta predicción:
    - GET: usa el dataset en data/raw/
    - POST: procesa archivo CSV subido por el usuario
    """
    try:
        if request.method == "POST":
            # =====================
            # PREDICCIÓN DESDE CSV SUBIDO
            # =====================
            if 'file' not in request.files:
                return jsonify({
                    "error": "No file provided",
                    "detail": "Debe enviar un archivo CSV con el campo 'file'"
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    "error": "No file selected",
                    "detail": "El archivo está vacío"
                }), 400
            
            if not file.filename.lower().endswith('.csv'):
                return jsonify({
                    "error": "Invalid file type",
                    "detail": "Solo se aceptan archivos CSV"
                }), 400
            
            # Leer el CSV
            content = file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
            
            # Detectar si las columnas están en español y mapear a inglés
            if 'monto' in df.columns or 'tarjeta_credito' in df.columns:
                df = df.rename(columns=COLUMN_MAPPING_ES_TO_EN)
            
            # Limitar a 500 transacciones para evitar sobrecarga
            df = df.head(500)
            
        else:
            # =====================
            # PREDICCIÓN DESDE DATASET LOCAL (GET)
            # =====================
            df = pipeline.load_and_prepare(dataset="test")
            df = df.head(500)

        predictions = []

        for idx, row in df.iterrows():
            raw_features = row.to_dict()

            is_fraud, prob, details = predictor.predict_fraud(raw_features)

            unix_ts = raw_features.get(Columns.UNIX_TIME) or raw_features.get('unix_time')
            fecha = None
            if pd.notna(unix_ts):
                try:
                    fecha = datetime.fromtimestamp(
                        int(unix_ts), tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    fecha = None

            # Obtener ID de transacción
            id_trans = raw_features.get('Unnamed: 0') or raw_features.get('id_transaccion') or idx

            predictions.append({
                "id_transaccion": id_trans,
                "es_fraude": bool(details["is_fraud"]),
                "probabilidad_fraude": round(float(details["probability"]), 4),
                "nivel_riesgo": details["risk_level"],
                "confianza_modelo": round(float(details["confidence"]), 4),

                # 🔹 Sprint 2 (CLASIFICACIÓN)
                "tipo_fraude": details.get("fraud_type"),
                "razones_fraude": details.get("fraud_reasons", []),

                "fecha": fecha,
                "monto": raw_features.get(Columns.AMT) or raw_features.get('amt'),
                "tarjeta_credito": raw_features.get(Columns.CC_NUM) or raw_features.get('cc_num'),
                "codigo_postal": raw_features.get('zip') or raw_features.get('codigo_postal'),
            })

        # Enriquecemos la transacción en español con los campos de predicción
        resultado = {
            **transaccion_es,
            "es_fraude": bool(details["is_fraud"]),
            "probabilidad_fraude": round(float(details["probability"]), 2),
            "nivel_riesgo": details["risk_level"],
            "umbral_decision": round(float(details["threshold"]), 2),
            "confianza_modelo": round(float(details["confidence"]), 2),
            "engineered_features": details.get("engineered_features", {}),
            "fecha": fecha
        }

        # Transacciones sospechosas
        sospechosas = [p for p in predictions if p["es_fraude"]]

    # Filas sospechosas completas (con todos los campos)
    sospechosas_completas = [
        row for row in predicciones
        if row["es_fraude"]
    ]

    # Versión simplificada solo con los campos que quieres mostrar
    """
        transacciones_sospechosas = [
            {
                "id_transaccion": row.get("id_transaccion"),
                "codigo_postal": row.get("codigo_postal"),
                "monto": row.get("monto"),
                "tarjeta_credito": row.get("tarjeta_credito"),
                "nivel_riesgo": row.get("nivel_riesgo"),
            }
            for row in sospechosas_completas
        ]
    """
    transacciones_sospechosas = []
        
    for row in sospechosas_completas:
        explanation = generate_explanation(
            row.get("engineered_features", {})
        )

        transacciones_sospechosas.append({
            "id_transaccion": row.get("id_transaccion"),
            "codigo_postal": row.get("codigo_postal"),
            "monto": row.get("monto"),
            "tarjeta_credito": row.get("tarjeta_credito"),
            "nivel_riesgo": row.get("nivel_riesgo"),
        response = {
            "total_transacciones": len(predictions),
            "total_sospechosas": len(sospechosas),
            "hay_alerta": len(sospechosas) > 0,
            "transacciones_sospechosas": sospechosas,
            "predicciones": predictions,  # Incluir todas las predicciones
        }

            # Sprint 3
            "tipo_fraude": explanation["tipo_fraude"],
            "explicacion": explanation["explicacion"],
            "razones": explanation["razones"],
        })


    # Puntaje global de fraude = probabilidad máxima de fraude
    puntaje_fraude = max(
        (row["probabilidad_fraude"] for row in predicciones),
        default=0.0
    )
        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            "error": "Prediction failed",
            "detail": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    # Ejecutar desde raíz:
    # python -m src.api.app
    app.run(debug=True)
