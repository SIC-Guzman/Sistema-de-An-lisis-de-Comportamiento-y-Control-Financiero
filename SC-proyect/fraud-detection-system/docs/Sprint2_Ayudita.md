# Guía de Inicio - Sprint 2

Tan solo es una idea de lo que viene y segun eso les hice la estructura. Obvio no debe ser una regla solo es una idea de como estructurar lo que viene. 

## Lo que ya está hecho (Sprint 1)
Está definido en el readme 

## Objetivos del Sprint 2

### Parte 1: Feature Engineering (Segun lo que ando entendiendo del sprint2)
1. Implementar features temporales
2. Implementar features geográficas
3. Implementar features de comportamiento
4. Implementar features de NLP

### Parte 2: Modelado (igual segun entiendo)
1. Entrenar modelo supervisado (XGBoost)
2. Entrenar modelo no supervisado (Isolation Forest)
3. Evaluar y comparar modelos
4. Guardar modelos entrenados

---

## Estructura de Archivos para Sprint 2

```
src/features/          # Aquí supuestamente trabajas PRINCIPALMENTE
├── __init__.py       # ya existe (vacío)
├── temporal.py       
├── geographic.py     
├── behavioral.py     
└── nlp.py           #estos ultimos los creas segun lo que hagas

src/models/           # Aquí trabajarás para MODELOS
├── __init__.py      # ya existe 
├── supervised_model.py    
└── anomaly_detector.py    

notebooks/            # Para experimentación
├── 01_exploratory_analysis.ipynb    
├── 02_feature_engineering.ipynb    
└── 03_model_training.ipynb         
```
Como dije no es nescesario seguir la idea esta. 

---

## Paso 1: Cargar Datos Procesados

Usa los datos ya procesados:

```python
from src.data.loader import DataLoader

# Cargar datos procesados
loader = DataLoader()

# Opción 1: Cargar desde archivos procesados
train_df = loader.load_data('data/processed/fraudTrain_processed.parquet')
test_df = loader.load_data('data/processed/fraudTest_processed.parquet')

# Los datos ya están limpios, validados y listos para usar
print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Fraud rate: {train_df['is_fraud'].mean():.4f}")
```

---


## Recursos Útiles

### Documentación de referencia:
- `docs/data_dictionary_es.md` - Descripción de todas las columnas
- `docs/cleaning_criteria_es.md` - Cómo se limpiaron los datos
- `config/config.yaml` - Configuración del sistema

### Constantes útiles:
```python
from src.utils.constants import (
    Columns,        # Nombres de columnas
    Features,       # Nombres de features
    Thresholds,     # Umbrales de detección
)
```

### Logging:
```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Mi mensaje")
logger.warning("Advertencia")
logger.error("Error")
```

---

## Consejos Importantes

1. **Usa los datos procesados** - No cargues de `data/raw/`, usa `data/processed/`

2. **Ordena por tiempo** - Los datos ya están ordenados, mantenlos así para features temporales

3. **Prueba con muestra** - Usa `--sample 0.1` para pruebas rápidas:
   ```python
   df = loader.load_data('...', sample_frac=0.1)
   ```

4. **Guarda progreso** - Guarda el dataset con features intermedias

5. **Documenta decisiones** - Agrega comentarios sobre por qué elegiste ciertos umbrales

6. **Revisa logs** - Los logs están en `logs/` para debugging

---

## Adiciones 

En los archivos que creé se puede ir escalando, añadiendo más funciones segun sea nescesario. Por el momento traté de tener todo en cuenta para no estar reincidiendo en el sprint 1. Cualquier cosa me comentas y lo vemos juntos, Suerte.