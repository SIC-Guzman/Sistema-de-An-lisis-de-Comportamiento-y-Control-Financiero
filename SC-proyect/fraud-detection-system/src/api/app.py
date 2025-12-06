# src/api/app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from pathlib import Path
from datetime import datetime

# IMPORTANTE:
# Usamos import relativo porque api y models están dentro de src
from src.models.predict import FraudPredictor

# ====== RUTAS BASE ======
# /src/api/app.py  -> parent = /src/api
# parent.parent    -> /src
# parent.parent.parent -> raíz del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]

# Modelo entrenado en: fraud-detection-system/models/fraud_model.pkl
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

# Carpeta de templates: fraud-detection-system/src/ui/templates
TEMPLATES_DIR = BASE_DIR / "src" / "ui" / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# Cargamos el modelo UNA sola vez
predictor = FraudPredictor(str(MODEL_PATH))


@app.route("/")
def index():
    # Renderiza src/ui/templates/index.html
    return render_template("index.html")


# Mapeo español -> columnas que espera el modelo
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

    # Copia en español (como viene el CSV)
    df_es = df.copy()

    # Renombrar PARA EL MODELO (inglés)
    df_model = df.rename(columns=COLUMN_MAPPING)

    registros = []

    for (_, row_model), (_, row_es) in zip(df_model.iterrows(), df_es.iterrows()):
        features = row_model.to_dict()

        # Predicción del modelo
        es_fraude, prob, details = predictor.predict_fraud(features)

        # Sacar fecha desde unix_time
        unix_ts = row_model.get("unix_time", None)
        fecha = None
        if pd.notna(unix_ts):
            try:
                fecha = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                fecha = None

        # Aquí SOLO guardamos lo que quieres + campos internos para lógica
        registros.append({
            "id_transaccion": row_es.get("id_transaccion"),
            "codigo_postal": row_es.get("codigo_postal"),
            "monto": row_es.get("monto"),
            "tarjeta_credito": row_es.get("tarjeta_credito"),
            "nivel_riesgo": details["risk_level"],
            "fecha": fecha,

            # internos (no los enviaremos al front)
            "_es_fraude": bool(details["is_fraud"]),
            "_probabilidad_fraude": float(details["probability"]),
        })

    # Filas sospechosas: SOLO con los campos visibles
    transacciones_sospechosas = [
        {
            "id_transaccion": r["id_transaccion"],
            "codigo_postal": r["codigo_postal"],
            "monto": r["monto"],
            "tarjeta_credito": r["tarjeta_credito"],
            "nivel_riesgo": r["nivel_riesgo"],
            "fecha": r["fecha"],
        }
        for r in registros
        if r["_es_fraude"]
    ]

    # Puntaje global de fraude (máxima probabilidad)
    puntaje_fraude = max(
        (r["_probabilidad_fraude"] for r in registros),
        default=0.0
    )

    response = {
        "hay_alerta": len(transacciones_sospechosas) > 0,
        "puntaje_fraude": puntaje_fraude,
        "total_transacciones": len(registros),
        "total_sospechosas": len(transacciones_sospechosas),
        "transacciones_sospechosas": transacciones_sospechosas,
    }

    return jsonify(response)

if __name__ == "__main__":
    # Ejecutar SIEMPRE desde la raíz del proyecto:
    #   python -m src.api.app
    app.run(debug=True)


