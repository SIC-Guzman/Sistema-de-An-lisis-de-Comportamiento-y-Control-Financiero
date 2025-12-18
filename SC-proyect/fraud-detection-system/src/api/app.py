from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pathlib import Path
from datetime import datetime, timezone
import io
import traceback

import pandas as pd

from src.models.mainPredictClass import FraudPredictor
from src.data.pipeline import DataPipeline
from src.utils.constants import Columns
from src.utils.explanations import generate_explanation

# =========================
# PATHS BASE
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.pkl"
TEMPLATES_DIR = BASE_DIR / "src" / "ui" / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
CORS(app)

# =========================
# MODELO Y PIPELINE
# =========================
predictor = FraudPredictor(str(MODEL_PATH))
pipeline = DataPipeline(BASE_DIR)

# =========================
# MAPEO COLUMNAS (ES -> EN)
# =========================
COLUMN_MAPPING_ES_TO_EN = {
    "id_transaccion": "Unnamed: 0",
    "monto": "amt",
    "tarjeta_credito": "cc_num",
    "poblacion_ciudad": "city_pop",
    "latitud_cliente": "lat",
    "longitud_cliente": "long",
    "latitud_comercio": "merch_lat",
    "longitud_comercio": "merch_long",
    "tiempo_unix": "unix_time",
    "codigo_postal": "zip",
}

# =========================
# HELPERS
# =========================
def _format_fecha(unix_ts) -> str | None:
    if unix_ts is None or pd.isna(unix_ts):
        return None
    try:
        return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _maybe_rename_to_english(df: pd.DataFrame) -> pd.DataFrame:
    # Si detectamos columnas ES, renombramos a EN
    es_cols_present = any(col in df.columns for col in COLUMN_MAPPING_ES_TO_EN.keys())
    if es_cols_present:
        df = df.rename(columns=COLUMN_MAPPING_ES_TO_EN)
    return df


def _build_prediction_row(idx: int, raw_features: dict, details: dict) -> dict:
    unix_ts = raw_features.get(getattr(Columns, "UNIX_TIME", "unix_time")) or raw_features.get("unix_time")
    fecha = _format_fecha(unix_ts)

    # ID transacción: preferimos Unnamed: 0 si existe
    id_trans = raw_features.get("Unnamed: 0")
    if id_trans is None or (isinstance(id_trans, float) and pd.isna(id_trans)):
        id_trans = raw_features.get("id_transaccion", idx)

    # Campos básicos de salida (en español)
    row_out = {
        "id_transaccion": id_trans,
        "es_fraude": bool(details.get("is_fraud", False)),
        "probabilidad_fraude": round(float(details.get("probability", 0.0)), 4),
        "nivel_riesgo": details.get("risk_level", "medio"),
        "confianza_modelo": round(float(details.get("confidence", 0.0)), 4),
        "umbral_decision": round(float(details.get("threshold", 0.5)), 4),

        "tipo_fraude": details.get("fraud_type"),
        "razones_fraude": details.get("fraud_reasons", []),

        "fecha": fecha,
        "monto": raw_features.get(getattr(Columns, "AMT", "amt")) or raw_features.get("amt"),
        "tarjeta_credito": raw_features.get(getattr(Columns, "CC_NUM", "cc_num")) or raw_features.get("cc_num"),
        "codigo_postal": raw_features.get("zip") or raw_features.get("codigo_postal"),
    }

    # Si tu predictor manda engineered_features, lo incluimos (sirve para explanations)
    if "engineered_features" in details:
        row_out["engineered_features"] = details["engineered_features"]

    return row_out


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    # si no tienes index.html, cambia por el template correcto
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  -> demo: usa dataset local desde pipeline
    POST -> recibe CSV (multipart/form-data) con campo 'file'
    """
    try:
        # =====================
        # 1) CARGA DATA
        # =====================
        if request.method == "POST":
            if "file" not in request.files:
                return jsonify({"error": True, "message": "Debe enviar un archivo CSV con el campo 'file'"}), 400

            file = request.files["file"]
            if not file.filename:
                return jsonify({"error": True, "message": "No se seleccionó ningún archivo"}), 400

            if not file.filename.lower().endswith(".csv"):
                return jsonify({"error": True, "message": "Solo se aceptan archivos CSV (.csv)"}), 400

            content = file.read().decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(content))
            df = _maybe_rename_to_english(df)

            # limite para demo / evitar sobrecarga
            df = df.head(500)

        else:
            df = pipeline.load_and_prepare(dataset="test").head(500)

        # =====================
        # 2) PREDICCIÓN
        # =====================
        predictions: list[dict] = []

        for idx, row in df.iterrows():
            raw_features = row.to_dict()
            _, _, details = predictor.predict_fraud(raw_features)
            pred_row = _build_prediction_row(idx, raw_features, details)
            predictions.append(pred_row)

        sospechosas = [p for p in predictions if p["es_fraude"]]

        # =====================
        # 3) TRANSACCIONES SOSPECHOSAS + EXPLICACIÓN
        # =====================
        transacciones_sospechosas = []
        for row in sospechosas:
            engineered = row.get("engineered_features", {})
            exp = generate_explanation(engineered) if engineered is not None else {
                "tipo_fraude": row.get("tipo_fraude"),
                "explicacion": "",
                "razones": row.get("razones_fraude", []),
            }

            transacciones_sospechosas.append({
                "id_transaccion": row.get("id_transaccion"),
                "codigo_postal": row.get("codigo_postal"),
                "monto": row.get("monto"),
                "tarjeta_credito": row.get("tarjeta_credito"),
                "nivel_riesgo": row.get("nivel_riesgo"),

                # Sprint 3 (explicabilidad)
                "tipo_fraude": exp.get("tipo_fraude"),
                "explicacion": exp.get("explicacion"),
                "razones": exp.get("razones"),

                "latitud_cliente":row.get("latitud_cliente"),
                "longitud_cliente":row.get("longitud_cliente"),
                "latitud_comercio":row.get("latitud_comercio"),
                "longitud_comercio":row.get("longitud_comercio")
            })

        # =====================
        # 4) RESPONSE
        # =====================
        puntaje_fraude = max((p["probabilidad_fraude"] for p in predictions), default=0.0)

        response = {
            "total_transacciones": len(predictions),
            "total_sospechosas": len(sospechosas),
            "hay_alerta": len(sospechosas) > 0,
            "puntaje_fraude": round(float(puntaje_fraude), 4),

            "transacciones_sospechosas": transacciones_sospechosas,
            "predicciones": predictions,
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": True,
            "message": "Prediction failed",
            "detail": str(e),
            "traceback": traceback.format_exc(),
        }), 500


if __name__ == "__main__":
    # Ejecutar desde raíz:
    # python -m src.api.app
    app.run(debug=True, host="0.0.0.0", port=5000)
