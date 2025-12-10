# Diccionario de Datos

## Resumen

Este documento proporciona una referencia completa para todos los campos del dataset Sparkov Credit Card Fraud Detection.

**Dataset**: Transacciones Sintéticas de Tarjetas de Crédito Sparkov
**Fuente**: https://www.kaggle.com/datasets/kartik2112/fraud-detection

---

## Información de Transacción

### `trans_date_trans_time`
- **Tipo**: datetime
- **Formato**: YYYY-MM-DD HH:MM:SS
- **Descripción**: Marca temporal cuando ocurrió la transacción
- **Ejemplo**: `2019-01-01 00:00:18`
- **Permite nulos**: No
- **Uso**: Feature temporal principal para análisis basado en tiempo

### `trans_num`
- **Tipo**: string
- **Descripción**: Identificador único de transacción
- **Ejemplo**: `0b242abb623afc578575680df30655b9`
- **Permite nulos**: No
- **Uso**: Llave única para transacciones

### `unix_time`
- **Tipo**: integer
- **Descripción**: Marca temporal de la transacción en formato Unix epoch
- **Ejemplo**: `1325376018`
- **Permite nulos**: No
- **Uso**: Representación alternativa de marca temporal

---

## Información de Monto

### `amt`
- **Tipo**: float
- **Rango**: 0 a ~30,000
- **Unidad**: USD ($)
- **Descripción**: Monto de la transacción en dólares
- **Ejemplo**: `4.97`
- **Permite nulos**: No
- **Resumen Estadístico**:
  - Media: ~67 USD
  - Mediana: ~47 USD
  - Desviación estándar: ~90 USD
- **Uso**: Feature principal para detección de anomalías basadas en monto

---

## Información del Titular

### `cc_num`
- **Tipo**: integer (16 dígitos)
- **Descripción**: Número de tarjeta de crédito (anonimizado pero consistente por usuario)
- **Ejemplo**: `2703186189652095`
- **Permite nulos**: No
- **Uso**: Identificador de usuario para análisis de comportamiento

### `first`
- **Tipo**: string
- **Descripción**: Nombre del titular
- **Ejemplo**: `Jennifer`
- **Permite nulos**: No
- **Uso**: Identificación de usuario (no se usa en modelado)

### `last`
- **Tipo**: string
- **Descripción**: Apellido del titular
- **Ejemplo**: `Banks`
- **Permite nulos**: No
- **Uso**: Identificación de usuario (no se usa en modelado)

### `gender`
- **Tipo**: categorical
- **Valores**: `M` (Masculino), `F` (Femenino)
- **Descripción**: Género del titular
- **Ejemplo**: `F`
- **Permite nulos**: No
- **Distribución**: División ~50/50
- **Uso**: Feature demográfico

### `dob`
- **Tipo**: date
- **Formato**: YYYY-MM-DD
- **Descripción**: Fecha de nacimiento del titular
- **Ejemplo**: `1988-03-09`
- **Permite nulos**: Sí (raro)
- **Uso**: Cálculo de edad para análisis demográfico

### `job`
- **Tipo**: string
- **Descripción**: Ocupación del titular
- **Ejemplo**: `Psychologist, counselling`
- **Permite nulos**: Sí (raro)
- **Categorías**: ~500 trabajos únicos
- **Uso**: Feature socioeconómico

---

## Información de Ubicación (Titular)

### `street`
- **Tipo**: string
- **Descripción**: Dirección de calle del titular
- **Ejemplo**: `561 Perry Cove`
- **Permite nulos**: No
- **Uso**: Referencia de dirección (no se usa en modelado)

### `city`
- **Tipo**: string
- **Descripción**: Ciudad del titular
- **Ejemplo**: `Moravian Falls`
- **Permite nulos**: No
- **Categorías**: ~994 ciudades únicas
- **Uso**: Análisis geográfico

### `state`
- **Tipo**: string (código de 2 letras)
- **Descripción**: Estado de EE.UU. del titular
- **Ejemplo**: `NC`
- **Permite nulos**: No
- **Valores**: Códigos estándar de estados de EE.UU.
- **Uso**: Análisis regional

### `zip`
- **Tipo**: integer (5 dígitos)
- **Descripción**: Código postal del titular
- **Ejemplo**: `28654`
- **Permite nulos**: No
- **Uso**: Análisis regional

### `lat`
- **Tipo**: float
- **Rango**: -90 a 90 grados
- **Descripción**: Latitud de la ubicación del titular
- **Ejemplo**: `36.0788`
- **Permite nulos**: No
- **Precisión**: 4 decimales (~11 metros)
- **Uso**: **Crítico para detección de fraude basada en distancia**

### `long`
- **Tipo**: float
- **Rango**: -180 a 180 grados
- **Descripción**: Longitud de la ubicación del titular
- **Ejemplo**: `-81.1781`
- **Permite nulos**: No
- **Precisión**: 4 decimales (~11 metros)
- **Uso**: **Crítico para detección de fraude basada en distancia**

### `city_pop`
- **Tipo**: integer
- **Descripción**: Población de la ciudad del titular
- **Ejemplo**: `3495`
- **Rango**: Cientos a millones
- **Permite nulos**: No
- **Uso**: Feature de urbanización

---

## Información del Comerciante

### `merchant`
- **Tipo**: string
- **Descripción**: Nombre del comerciante donde ocurrió la transacción
- **Ejemplo**: `fraud_Rippin, Kub and Mann`
- **Permite nulos**: No
- **Patrón**: Frecuentemente con prefijo `fraud_` para transacciones fraudulentas
- **Categorías**: ~693 comerciantes únicos
- **Uso**: **Detección de patrones basada en comerciante**

### `category`
- **Tipo**: categorical
- **Descripción**: Categoría del comerciante/transacción
- **Ejemplo**: `misc_net`
- **Permite nulos**: No
- **Valores**: 14 categorías
  - `gas_transport` - Gasolineras y transporte
  - `grocery_pos` - Tiendas de abarrotes
  - `home` - Mejoras para el hogar
  - `misc_net` - Misceláneos en línea
  - `misc_pos` - Misceláneos punto de venta
  - `shopping_net` - Compras en línea
  - `shopping_pos` - Compras en tienda
  - `entertainment` - Lugares de entretenimiento
  - `food_dining` - Restaurantes y comida
  - `personal_care` - Servicios de cuidado personal
  - `health_fitness` - Salud y fitness
  - `travel` - Relacionado con viajes
  - `kids_pets` - Artículos para niños y mascotas
- **Distribución**: Varía por categoría
- **Uso**: **Análisis de patrones basado en categoría**

### `merch_lat`
- **Tipo**: float
- **Rango**: -90 a 90 grados
- **Descripción**: Latitud de la ubicación del comerciante
- **Ejemplo**: `36.011293`
- **Permite nulos**: Raro
- **Uso**: Cálculo de distancia desde el titular

### `merch_long`
- **Tipo**: float
- **Rango**: -180 a 180 grados
- **Descripción**: Longitud de la ubicación del comerciante
- **Ejemplo**: `-82.048315`
- **Permite nulos**: Raro
- **Uso**: Cálculo de distancia desde el titular

---

## Variable Objetivo

### `is_fraud`
- **Tipo**: integer binario
- **Valores**: 
  - `0` - Transacción legítima
  - `1` - Transacción fraudulenta
- **Descripción**: Etiqueta verdadera de fraude
- **Ejemplo**: `0`
- **Permite nulos**: No
- **Distribución**: 
  - Entrenamiento: ~0.58% fraude
  - Prueba: ~0.39% fraude
- **Uso**: **Variable objetivo para aprendizaje supervisado**

---

## Features Derivadas (Creadas en Ingeniería de Características)

E(stas son importantes para el sprint2)

### Features Temporales
- `hour` - Hora del día (0-23)
- `day_of_week` - Día de la semana (0-6)
- `month` - Mes (1-12)
- `is_weekend` - Indicador booleano
- `is_night` - Booleano para 2 AM - 6 AM
- `time_since_last_trans` - Segundos desde última transacción
- `trans_velocity` - Transacciones por hora/día

### Features Geográficas
- `distance_from_home` - Distancia entre comerciante y titular (km)
- `distance_from_last` - Distancia desde transacción anterior (km)
- `geographic_velocity` - km/h entre transacciones
- `location_entropy` - Diversidad de ubicaciones usadas

### Features de Comportamiento
- `amt_mean_7d` - Monto promedio últimos 7 días
- `amt_std_7d` - Desviación estándar de monto últimos 7 días
- `trans_freq_7d` - Conteo de transacciones últimos 7 días
- `merchant_diversity` - Número de comerciantes únicos
- `category_entropy` - Diversidad de categorías usadas

### Features de NLP
- `merchant_risk_score` - Puntuación de riesgo basada en nombre del comerciante
- `has_suspicious_keyword` - Booleano para términos sospechosos

---

## Notas de Calidad de Datos

### Valores Faltantes
- **Muy Raros**: <0.01% del total de datos
- **Columnas Afectadas**: Principalmente `job`, `dob`
- **Manejo**: Ver cleaning_criteria.md

### Integridad de Datos
- **Duplicados**: Ninguno encontrado (trans_num único)
- **Consistencia**: Datos sintéticos de alta calidad
- **Ordenamiento Temporal**: Ordenado cronológicamente

### Problemas Conocidos
- Algunos nombres de comerciantes tienen prefijo "fraud_" en transacciones fraudulentas (fuga de datos, no debe usarse como feature)
- Desajustes ocasionales de coordenadas (raro)

---

## Ejemplos de Uso

### Cargar e Inspeccionar Datos
```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_data('data/raw/fraudTrain.csv')

# Ver información básica
print(df.head())
print(df.info())
print(df.describe())
```

### Acceder a Columnas Específicas
```python
from src.utils.constants import Columns

# Análisis temporal
transactions_by_hour = df.groupby(df[Columns.TRANS_DATE_TIME].dt.hour).size()

# Análisis geográfico
import folium
map = folium.Map(location=[df[Columns.LAT].mean(), df[Columns.LONG].mean()])
```

### Análisis de Fraude
```python
# Tasa de fraude por categoría
fraud_by_category = df.groupby(Columns.CATEGORY)[Columns.IS_FRAUD].mean()

# Distribución de montos
fraud_amt = df[df[Columns.IS_FRAUD] == 1][Columns.AMT]
legit_amt = df[df[Columns.IS_FRAUD] == 0][Columns.AMT]
```

---

