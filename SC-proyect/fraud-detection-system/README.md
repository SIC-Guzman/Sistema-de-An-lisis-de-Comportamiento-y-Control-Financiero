# Control Financiero - Sistema de Análisis de Comportamiento

Sistema de análisis de comportamiento financiero diseñado para detectar transacciones fraudulentas mediante Machine Learning, análisis geográfico y procesamiento de lenguaje natural.

---

## Descripción del Proyecto

Este proyecto implementa un sistema completo de detección de fraude que analiza transacciones financieras para identificar comportamientos anómalos mediante:

- **Análisis temporal**: Detección de patrones inusuales de horario y frecuencia
- **Análisis geográfico**: Identificación de cambios bruscos de ubicación y viajes imposibles
- **Análisis de comportamiento**: Evaluación de patrones de gasto y hábitos del usuario
- **Procesamiento de texto**: Clasificación de comerciantes y detección de términos sospechosos

---

## Tecnologías Utilizadas

### Lenguaje y Frameworks
- **Python 3.9+**: Lenguaje principal
- **pandas**: Manipulación y análisis de datos
- **scikit-learn**: Modelos de Machine Learning
- **XGBoost**: Algoritmo de clasificación avanzado

### Librerías Especializadas
- **geopy**: Cálculos geográficos y distancias
- **NLTK/spaCy**: Procesamiento de lenguaje natural
- **FastAPI**: API REST (Sprint 3)
- **Streamlit**: Dashboard interactivo (Sprint 3)

### Herramientas de Desarrollo
- **pytest**: Testing unitario
- **YAML**: Configuración del sistema
- **Git**: Control de versiones

---

## Dataset

**Fuente**: Sparkov Credit Card Fraud Detection Dataset (Kaggle)
- **URL**: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Tamaño**: 1.8 millones de transacciones sintéticas
- **Período**: 18 meses de datos
- **Características**: 22 columnas incluyendo coordenadas geográficas
- **Tasa de fraude**: ~0.5% (distribución realista)
- **Licencia**: CC0 (Dominio Público)

El dataset incluye información temporal, geográfica, demográfica y transaccional, ideal para entrenar modelos robustos de detección de fraude.

---

## Estructura del Proyecto

```
fraud-detection-system/
│
├── data/                          # Datos del proyecto
│   ├── raw/                       # Datos originales de Kaggle
│   ├── processed/                 # Datos procesados y limpios
│   ├── interim/                   # Datos intermedios
│   └── external/                  # Datos de referencia externa
│
├── src/                           # Código fuente
│   ├── data/                      # Módulos de datos
│   │   ├── loader.py             # Carga de datos
│   │   ├── cleaner.py            # Limpieza de datos
│   │   └── schema.py             # Validación de esquemas
│   │
│   ├── features/                  # Ingeniería de características
│   │   ├── temporal.py           # Features temporales
│   │   ├── geographic.py         # Features geográficas
│   │   ├── behavioral.py         # Features de comportamiento
│   │   └── nlp.py                # Features de texto
│   │
│   ├── models/                    # Modelos de ML
│   │   ├── anomaly_detector.py   # Detector de anomalías
│   │   └── supervised_model.py   # Modelo supervisado
│   │
│   ├── api/                       # API REST
│   │   └── endpoints.py          # Endpoints de la API
│   │
│   ├── ui/                        # Interfaz de usuario
│   │   └── dashboard.py          # Dashboard Streamlit
│   │
│   └── utils/                     # Utilidades
│       ├── config.py             # Gestión de configuración
│       ├── logger.py             # Sistema de logging
│       └── constants.py          # Constantes del sistema
│
├── tests/                         # Tests unitarios
│   ├── test_data/                # Tests de módulos de datos
│   └── test_features/            # Tests de features
│
├── notebooks/                     # Jupyter notebooks
│
├── models/                        # Modelos entrenados
├── reports/                       # Reportes y visualizaciones
├── config/                        # Archivos de configuración
│   └── config.yaml               # Configuración principal
│
├── scripts/                       # Scripts de utilidad
│   ├── download_dataset.py       # Descarga de datos
│   └── preprocess_pipeline.py    # Pipeline de preprocesamiento
│
├── docs/                          # Documentación
│   ├── data_dictionary.md        # Diccionario de datos
│   └── cleaning_criteria.md      # Criterios de limpieza
│
├── requirements.txt               # Dependencias Python
├── setup.py                       # Configuración de instalación
└── README.md                      # Este archivo
```

---

## Instalación [para verificar el sprint1 (al momento)]

### Prerrequisitos

- Python 3.9 o superior
- pip 

### Pasos de Instalación

1. **Descargar proyecto**
   ```bash
   # Descomprime y navega a la carpeta
   cd fraud-detection-system
   ```

2. **Crear ambiente virtual**
   
   Para aislar las dependencias del proyecto:
   
   ```bash
   # Crear ambiente virtual
   python -m venv venv
   ```

3. **Activar el ambiente virtual**
   
   ```bash
   # En Windows:
   venv\Scripts\activate
   
   # En Mac/Linux:
   source venv/bin/activate
   ```
   


4. **Instalar dependencias**
   
   ```bash
   # Instalación de todas las librerías necesarias 
   pip install -r requirements.txt
   ```

5. **Verificar instalación**
   
   ```bash
   # Verificar que las librerías principales estén instaladas
   python -c "import pandas; import sklearn; print('Instalación exitosa')"
   ```

---

## Configuración de Datos

### Datos ya incluidos (si no, esta en un zip o me lo piden o lo descargan)
Los datos ya están en la carpeta `data/raw/`



---

## Uso del Sistema

### Pipeline de Preprocesamiento

El pipeline de preprocesamiento limpia y valida los datos automáticamente.

**Ejecutar pipeline completo:**
```bash
# Como los datos ya están en data/raw/
python scripts/preprocess_pipeline.py --skip-download
# eso de skip es porq lo tenia como una descarga que se podia actualizar, pero mejor me quede con lo de los datos descargados previamente para mejor manejo
python -m src.models.predict --mode train --model_path models/fraud_model.pkl
python -m src.models.predecir --mode train --model_path models/fraud_model.pkl
```

**Opciones disponibles:**
```bash
# Usar una muestra del 10% para pruebas rápidas
python scripts/preprocess_pipeline.py --sample 0.1

# Solo validar datos sin procesar
python scripts/preprocess_pipeline.py --validate-only

# Guardar en formato Parquet o CSV
python scripts/preprocess_pipeline.py --output-format parquet
```

**Salida esperada:**
```
================================================================
PREPROCESSING SUMMARY
================================================================

Training Set:
  Rows: 1,296,675
  Fraud cases: 7,506 (0.58%)
  
Test Set:
  Rows: 555,719
  Fraud cases: 2,145 (0.39%)
  
Data Quality:
  Missing values: 0
  Duplicates: 0
  
Preprocessing pipeline complete!
```

**Archivos generados:**
- `data/processed/fraudTrain_processed.parquet` - Datos de entrenamiento limpios
- `data/processed/fraudTest_processed.parquet` - Datos de prueba limpios
- `logs/fraud_detection_YYYYMMDD_HHMMSS.log` - Log detallado del proceso

---

## Ejecución de Tests

Para verificar que los módulos funcionan correctamente:

```bash
# Ejecutar todos los tests
pytest

# Ejecutar tests con cobertura
pytest --cov=src tests/

# Ejecutar tests específicos
pytest tests/test_data/test_loader.py
```


---

## Desarrollo de Frontend 

Si el front lo hacen en React. Como idea abstracta para el que le toque: 

### React Frontend

```
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx
│   │   ├── TransactionMap.jsx
│   │   ├── AlertPanel.jsx
│   │   └── Timeline.jsx
│   ├── services/
│   │   └── api.js          # Llamadas a la API
│   └── App.jsx
├── package.json
└── README.md
```

**Integración con API:**
```bash
# Ubicación de la API en src/api/app.py

# activar tu venv si no esta habilitado
  # En Windows:
  venv\Scripts\activate
  
  # En Mac/Linux:
  source venv/bin/activate

# ejecución desde la carpeta raiz (SC-proyect/fraud-detection-system)
python -m src.api.app
API_BASE_URL = 'http://127.0.0.1:5000';

```

---

## Configuración

El sistema utiliza configuración centralizada en `config/config.yaml`.

**Principales configuraciones:**

```yaml
# Rutas de datos
paths:
  data:
    raw: "data/raw"
    processed: "data/processed"

# Limpieza de datos
cleaning:
  remove_duplicates: true
  handle_missing:
    strategy: "drop"

# Features
features:
  temporal:
    extract_hour: true
    extract_day_of_week: true
  geographic:
    calculate_distance_from_home: true
    max_reasonable_velocity: 800  # km/h

# Modelos
model:
  supervised:
    algorithm: "xgboost"
  unsupervised:
    algorithm: "isolation_forest"
```

**Modificar configuración:**
1. Editar `config/config.yaml`
2. Los cambios se aplican automáticamente al reiniciar

---

## Documentación Adicional

### Referencia de Datos

Para los que tengan que utilizar los datos les dejo una idea de como tratarlos.`docs/data_dictionary.md` para descripción detallada de:
- Todos los campos del dataset
- Tipos de datos y rangos válidos
- Ejemplos de uso
- Features derivadas

### Criterios de Limpieza

`docs/cleaning_criteria.md` para entender:
- Reglas de limpieza aplicadas
- Manejo de valores faltantes
- Validación de restricciones
- Detección de outliers (sin eliminación)

---

---

## Licencia

Este proyecto utiliza el dataset Sparkov bajo licencia CC0 (Dominio Público).

El código del proyecto está disponible para uso académico.

---

## Equipo

- **Sprint 1 (Datos y Preprocesamiento)**: [Eric David Rojas de León]
- **Sprint 2 (Features y Modelos)**: [Nombre]
- **Sprint 3 (API y UI)**: [Nombre]
- **Sprint 4 (Testing y Deploy)**: [Nombre]

---

## Grupo 3
### Samsung Innovation Campus
