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

### Flujo de ejecucion

1. **Datos csv**
   - Debes descargar el archivo zip del sigueinte link: [kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
   - Luego descomprimir la descarga y agregar los archivos llamados: [fraudText.csv] y [fraudTrain.csv] a la ruta /data/raw/

2. **Preprocesar datos**
   
   ```bash
   python scripts/preprocess_pipeline.py --skip-download
   ```

3. **Entrenar el modelo Random Forest**
   
   ```bash
   python src/models/XGBoost.py
   ```

4. **Probar predicción local**
   
   ```bash
   python src/models/predecir.py
   ```

5. **Levantar backend (API)**
   
   ```bash
   python -m src.api.app
   ```

6. **Levantar interfaz gráfica (UI)**
   
   ```bash
   python src/ui/app.py
   ```
---

## Ejemplo de dataset valido

```
amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,type
5000,10000,5000,20000,25000,TRANSFER
1500,3000,1500,0,1500,PAYMENT
10000,50000,40000,120000,130000,CASH_OUT
```


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
## Organización y explicación de scripts para el modelo
### Organización
Dentro de la subcarpeta *models* de la carpeta src se encuentran 3 archivos:

```
src/
├── api/
├── data/
├── features/
├── models/
│   ├── predecir.py/
│   ├── mainPredictClass.py/
│   ├── randomForest.py/
├── ui/
├── utils/
```
La funcionalidad de cada uno de los archivos indicados anteriormente se encuentra a contuación:
- **randomFores.py:** Archivo con la estructura del modelo, cuenta con las configuraciones del modelo y genera el archivo *fraud_model.pkl* (El modelo ya entrenado) para su uso en los archivos de predicción.
- **mainPredictClass.py:** Cuenta con la clase principal que se encarga de predecir resultados tomando parametros de entrada. Adicionalmente, cuenta con algunos ejemplos de predicción
- **predecir.py:** Cuenta con ejemplos variados de predicción haciendo uso de la clase *FraudPredictor* del archivo *predict_base.py*

La ejecución y aprovechamiento del código es simple:
1. Ejecutar el archivo *randomForest.py* para generar el archivo .pkl del modelo.
2. Ejecutar el archivo *predecir.py* que aprovecha la clase de mainPredictClass para hacer predicciones de acuerdo a parametros de entrada para observar el resultado de los ejemplos.

*Nota: Siempre ejecutar todo dentro del directorio principal: "\fraud-detection-system"*

---

## Desarrollo de Frontend 

### Flask Frontend

```
frontend/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── loader.py
│   │   └── schema.py      
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mainPredictClass.py
│   │   ├── predecir.py
│   │   └── XGBoost.py
│   ├── ui/
│   │   ├── static/
│   │   |   └── schema.py  
│   │   ├── templates/
│   │   |   |── dashboard.html  
│   │   |   |── index.html  
│   │   |   |── layout.html 
│   │   |   |── manual_predict.html
│   │   |   └── predict.html
│   │   ├── __init__.py
│   │   └── app.py
│   └── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   └── logger.py    
└── __init__.py
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

Idea de como tratar los datos. `docs/data_dictionary.md` para descripción detallada de:
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
- **Sprint 2 (Features y Modelos)**: [Camilo Ernesto Sincal Sipac]
- **Sprint 3 (API y UI)**: [Karen Michelle Gatica Arriola ] y [Génesis Paola Gómez Fernández]
- **Sprint 4 (Testing y Deploy)**: [Alberto Moisés Gerardo Lémus Alvarado]

---

## Grupo 2
### Samsung Innovation Campus
