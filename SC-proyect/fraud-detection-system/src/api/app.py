from flask import Flask, request, jsonify, render_template
from src.utils.explanations import generate_explanation
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone 

# IMPORTANTE:
# Usamos import relativo porque api y models están dentro de src
from src.models.mainPredictClass import FraudPredictor

# ====== RUTAS BASE ======
# /src/api/app.py  -> parent = /src/api
# parent.parent    -> /src
# parent.parent.parent -> raíz del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]

# Modelo entrenado en: fraud-detection-system/models/xgboost_fraud_model.pkl
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.pkl"

# Carpeta de templates: fraud-detection-system/src/ui/templates
TEMPLATES_DIR = BASE_DIR / "src" / "ui" / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

@app.route("/")
def index():
    return render_template("index.html")


# Cargamos el modelo UNA sola vez
predictor = FraudPredictor(str(MODEL_PATH))

# Mapeo de columnas del CSV en español -> columnas que espera el modelo
COLUMN_MAPPING = {
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

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if file is None:
        return jsonify({"error": 'No se envió ningún archivo con la clave "file"'}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error al leer el CSV: {e}"}), 400

    # Copia en español para devolver al front
    df_es = df.copy()

    # Renombrar para el modelo (internamente en inglés)
    df_model = df.rename(columns=COLUMN_MAPPING)

    predicciones = []

    # Recorremos filas: modelo (inglés) vs respuesta (español)
    for (_, row_model), (_, row_es) in zip(df_model.iterrows(), df_es.iterrows()):
        features_model = row_model.to_dict()
        transaccion_es = row_es.to_dict()

        # Predicción del modelo
        is_fraud, prob, details = predictor.predict_fraud(features_model)

         # --- FORMATEO CORRECTO DE FECHA ---
        unix_ts = transaccion_es.get("tiempo_unix")

        fecha = None
        if pd.notna(unix_ts):
            try:
                fecha = datetime.fromtimestamp(
                    int(unix_ts), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
            except:
                fecha = None


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

        predicciones.append(resultado)

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

    response = {
        "puntaje_fraude": puntaje_fraude,
        "hay_alerta": len(transacciones_sospechosas) > 0,
        "total_transacciones": len(predicciones),
        "total_sospechosas": len(transacciones_sospechosas),
        "transacciones_sospechosas": transacciones_sospechosas,
        # Si el profe quiere ver todo, acá sigue estando todo el detalle:
        "predicciones": predicciones
    }

    return jsonify(response)


if __name__ == "__main__":
    # Ejecutar SIEMPRE desde la raíz del proyecto:
    #   python -m src.api.app
    app.run(debug=True) 