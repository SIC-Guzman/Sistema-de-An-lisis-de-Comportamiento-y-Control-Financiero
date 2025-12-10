# Criterios de Limpieza de Datos

## Resumen

para este documento detallo todos los procedimientos, reglas , criterios de decisión para la limpieza de datos del sistema. Estas reglas aseguran la calidad, consistencia y preparación de los datos para machine learning.

---

## Resumen del Pipeline de Limpieza

El pipeline de limpieza ejecuta en este orden:

1. **Eliminación de Duplicados**
2. **Manejo de Valores Faltantes**
3. **Normalización de Tipos de Datos**
4. **Validación de Restricciones**
5. **Detección de Outliers** (opcional)
6. **Ordenamiento Temporal**

---

## 1. Eliminación de Duplicados

### Criterios

**Duplicados Exactos**:
- Eliminar filas donde TODAS las columnas sean idénticas
- Mantener: Primera ocurrencia
- Eliminar: Todas las ocurrencias subsecuentes

**Duplicados por ID de Transacción**:
- Eliminar filas con `trans_num` duplicado
- Mantener: Primera ocurrencia
- Eliminar: Todas las ocurrencias subsecuentes

### Justificación

- Cada transacción debe ser única
- `trans_num` está diseñado para ser un identificador único
- Los duplicados podrían inflar artificialmente el rendimiento del modelo

### Implementación

```python
# Eliminar duplicados exactos
df = df.drop_duplicates()

# Eliminar duplicados por número de transacción
df = df.drop_duplicates(subset=['trans_num'], keep='first')
```

### Estadísticas Rastreadas

- `duplicates_removed`: Cantidad de filas duplicadas eliminadas

---

## 2. Manejo de Valores Faltantes

### Estrategia por Importancia de Columna

#### **Columnas Críticas** (Eliminar fila si falta)

Estas columnas son esenciales para la detección de fraude:

| Columna | Justificación |
|---------|---------------|
| `trans_date_trans_time` | Requerida para todo análisis temporal |
| `cc_num` | Requerida para análisis de comportamiento del usuario |
| `amt` | Feature principal - no se puede imputar montos de transacción |
| `lat` | Requerida para detección de fraude geográfico |
| `long` | Requerida para detección de fraude geográfico |
| `is_fraud` | Variable objetivo - no puede faltar |

**Acción**: Eliminar toda la fila si alguna de estas falta

#### **Columnas Categóricas** (Rellenar con 'Unknown')

| Columna | Valor de Relleno | Justificación |
|---------|------------------|---------------|
| `category` | 'Unknown' | Preserva la fila, 'Unknown' se convierte en una categoría |
| `merchant` | 'Unknown' | Mejor que eliminar la transacción |
| `gender` | 'Unknown' | Feature demográfico menor |
| `job` | 'Unknown' | Feature demográfico opcional |

**Acción**: Rellenar con cadena 'Unknown'

#### **Columnas Numéricas** (Rellenar con mediana)

| Columna | Estrategia de Relleno | Justificación |
|---------|----------------------|---------------|
| `city_pop` | Mediana | La mediana es robusta ante outliers |

**Acción**: Rellenar con la mediana de la columna

#### **Coordenadas Geográficas** (Manejo especial)

| Columna | Estrategia de Relleno | Justificación |
|---------|----------------------|---------------|
| `merch_lat` | Copiar de `lat` | Asumir comerciante en ubicación del usuario |
| `merch_long` | Copiar de `long` | Suposición conservadora |

**Acción**: Usar coordenadas del usuario como respaldo

### Umbrales

- **Proporción Máxima Faltante**: 5% por columna
- Si >5% falta en columna crítica → investigar antes de procesar
- Si >20% falta en cualquier columna → marcar para revisión

### Implementación

```python
# Columnas críticas
critical = ['trans_date_trans_time', 'cc_num', 'amt', 'lat', 'long', 'is_fraud']
df = df.dropna(subset=critical)

# Columnas categóricas
categorical = ['category', 'merchant', 'gender', 'job']
for col in categorical:
    df[col] = df[col].fillna('Unknown')

# Columnas numéricas
df['city_pop'] = df['city_pop'].fillna(df['city_pop'].median())

# Geográficas
df['merch_lat'] = df['merch_lat'].fillna(df['lat'])
df['merch_long'] = df['merch_long'].fillna(df['long'])
```

---

## 3. Normalización de Tipos de Datos

### Columnas de Fecha/Hora

| Columna | Tipo Objetivo | Formato |
|---------|--------------|---------|
| `trans_date_trans_time` | datetime64[ns] | YYYY-MM-DD HH:MM:SS |
| `dob` | datetime64[ns] | YYYY-MM-DD |

**Manejo de Errores**: 
- Usar `errors='coerce'` para convertir fechas inválidas a NaT
- NaT en `trans_date_trans_time` → eliminar fila (campo crítico)
- NaT en `dob` → aceptable (campo opcional)

### Columnas Numéricas

| Columna | Tipo Objetivo | Justificación |
|---------|--------------|---------------|
| `amt` | float64 | Se necesita precisión decimal |
| `lat` | float64 | Precisión de coordenadas |
| `long` | float64 | Precisión de coordenadas |
| `merch_lat` | float64 | Precisión de coordenadas |
| `merch_long` | float64 | Precisión de coordenadas |
| `city_pop` | int64 | La población es un entero |
| `unix_time` | int64 | El timestamp es un entero |
| `is_fraud` | int64 | Binario: 0 o 1 |

**Manejo de Errores**:
- Usar `errors='coerce'` para convertir valores inválidos a NaN
- NaN en campos numéricos críticos → eliminar fila

### Columnas de Texto/Categóricas

Todas las columnas restantes se convierten a tipo `str`:
- `first`, `last`, `gender`, `street`, `city`, `state`, `zip`
- `job`, `merchant`, `category`, `trans_num`

### Implementación

```python
# Fecha/hora
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

# Numéricas
numeric_cols = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 'unix_time']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Etiqueta de fraude
df['is_fraud'] = df['is_fraud'].astype(int)

# Categóricas
categorical_cols = ['merchant', 'category', 'gender', 'trans_num', ...]
for col in categorical_cols:
    df[col] = df[col].astype(str)
```

---

## 4. Validación de Restricciones

### Restricciones de Monto

| Restricción | Valor | Acción |
|------------|-------|--------|
| Mínimo | 0.00 | Eliminar filas con amt < 0 |
| Máximo | 100,000.00 | Eliminar filas con amt > 100,000 |

**Justificación**:
- Montos negativos son errores de datos (no son reembolsos en este dataset)
- $100K es un límite superior razonable; valores mayores probablemente son errores
- Preserva el 99.9%+ de los datos

### Restricciones Geográficas

**Latitud**:
- Rango válido: -90° a 90°
- Eliminar filas fuera del rango

**Longitud**:
- Rango válido: -180° a 180°
- Eliminar filas fuera del rango

**Justificación**: Coordenadas inválidas impiden el análisis geográfico

### Restricciones de Género

- Valores válidos: 'M', 'F', 'Unknown'
- Eliminar filas con otros valores (errores de datos raros)

### Restricciones de Etiqueta de Fraude

- Valores válidos: 0, 1
- Eliminar filas con otros valores (campo crítico)

### Implementación

```python
# Monto
df = df[(df['amt'] >= 0) & (df['amt'] <= 100000)]

# Latitud
df = df[(df['lat'] >= -90) & (df['lat'] <= 90)]

# Longitud
df = df[(df['long'] >= -180) & (df['long'] <= 180)]

# Género
df = df[df['gender'].isin(['M', 'F', 'Unknown'])]

# Etiqueta de fraude
df = df[df['is_fraud'].isin([0, 1])]
```

---

## 5. Detección de Outliers

### Estrategia

**Método**: RIC (Rango Intercuartílico / IQR)
- Q1 = Percentil 25
- Q3 = Percentil 75
- RIC = Q3 - Q1
- Límite inferior = Q1 - 3.0 * RIC
- Límite superior = Q3 + 3.0 * RIC

### Columnas Analizadas

Actualmente solo se analiza **`amt`** (monto):
- Otras columnas tienen límites naturales (lat/long, categóricas, etc.)
- Los outliers de monto son señales importantes de fraude

### **Acción**: SOLO REGISTRAR, NO ELIMINAR

**Decisión Crítica**: NO eliminamos outliers porque:
1. El fraude frecuentemente involucra montos inusuales
2. Eliminar outliers eliminaría potenciales señales de fraude
3. Los modelos deben aprender de valores extremos

### Método Alternativo

**Método de Z-Score** (disponible pero no predeterminado):
- Calcular: z = (x - media) / desviación
- Umbral: |z| > 3.0
- Misma acción: solo registrar

### Implementación

```python
# Método RIC
Q1 = df['amt'].quantile(0.25)
Q3 = df['amt'].quantile(0.75)
RIC = Q3 - Q1

limite_inferior = Q1 - 3.0 * RIC
limite_superior = Q3 + 3.0 * RIC

outliers = (df['amt'] < limite_inferior) | (df['amt'] > limite_superior)
print(f"Se encontraron {outliers.sum()} outliers (no eliminados)")
```

---

## 6. Ordenamiento Temporal

### Paso Final

Ordenar todo el dataset por `trans_date_trans_time` en orden ascendente:
- Asegura orden cronológico
- Crítico para features de series temporales
- Permite cálculos eficientes de ventanas móviles

### Implementación

```python
df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
```

---

## Directrices de Retención de Datos

### Pérdida de Datos Esperada

| Etapa | Pérdida Esperada | Rango Aceptable |
|-------|-----------------|-----------------|
| Duplicados | <0.01% | 0-0.1% |
| Faltantes (críticos) | <0.01% | 0-0.1% |
| Violaciones de restricciones | <0.01% | 0-0.1% |
| **Total** | **<0.1%** | **0-0.5%** |

### Compuertas de Calidad

**DETENER Procesamiento Si**:
- Pérdida total de datos >5%
- La tasa de fraude cambia en >50% (ej., 0.5% → 0.25% o 0.75%)
- Cualquier columna crítica tiene >5% faltante
- La conversión de tipo de datos falla para >1% de filas

**Investigar y Ajustar Si**:
- Pérdida de datos >0.5%
- Una restricción específica elimina >100 filas
- La distribución de fraude se altera significativamente

---

## Verificaciones de Validación

### Validación Pre-Limpieza

```python
# Verificar datos originales
print(f"Forma original: {df.shape}")
print(f"Valores faltantes:\n{df.isnull().sum()}")
print(f"Duplicados: {df.duplicated().sum()}")
print(f"Tasa de fraude: {df['is_fraud'].mean():.4f}")
```

### Validación Post-Limpieza

```python
# Verificar datos limpios
print(f"Forma limpia: {df_clean.shape}")
print(f"Retención de datos: {len(df_clean)/len(df)*100:.2f}%")
print(f"Valores faltantes: {df_clean.isnull().sum().sum()}")
print(f"Tasa de fraude: {df_clean['is_fraud'].mean():.4f}")

# Validar restricciones
assert df_clean['amt'].min() >= 0
assert df_clean['amt'].max() <= 100000
assert df_clean['lat'].between(-90, 90).all()
assert df_clean['long'].between(-180, 180).all()
assert df_clean['is_fraud'].isin([0, 1]).all()
```

