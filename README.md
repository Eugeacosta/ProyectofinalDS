# 🛠️ Análisis de Ventas y Predicción con Machine Learning - Ferretería Retail

Este proyecto presenta un flujo de trabajo completo de Ciencia de Datos aplicado a un dataset de ventas minoristas de una cadena de ferreterías. El objetivo abarca desde la limpieza de datos y el análisis exploratorio (EDA) hasta la implementación de un modelo de Machine Learning para predecir el total de ventas.

## 📋 Tabla de Contenidos
- [Contexto del Proyecto](#-contexto-del-proyecto)
- [Dataset y Variables](#-dataset-y-variables)
- [Metodología](#-metodología)
- [Hallazgos del EDA](#-hallazgos-del-eda)
- [Modelo de Machine Learning](#-modelo-de-machine-learning)
- [Requisitos y Ejecución](#-requisitos-y-ejecución)
- [Autor](#-Maria-Eugenia-Acosta)

## 📖 Contexto del Proyecto
En un mercado competitivo, entender los patrones de compra es vital. Este proyecto busca responder preguntas clave de negocio:
- ¿Qué categorías generan más ingresos?
- ¿Cómo influyen los medios de pago y las promociones?
- ¿Es posible predecir el valor de una venta basándose en las características de la transacción?

El análisis está orientado a proveer *insights* accionables para la Gerencia General, Sucursales y Marketing.

## 💾 Dataset y Variables
El dataset `retail_ferreteria.csv` contiene transacciones históricas con las siguientes características principales:

| Variable | Descripción |
| :--- | :--- |
| `id_venta` / `id_cliente` | Identificadores únicos. |
| `sucursal` / `provincia` | Ubicación geográfica de la venta. |
| `categoria` | Tipo de producto (ej. Herramientas, Iluminación). |
| `precio_unitario` | Precio por unidad. |
| `cantidad` | Unidades vendidas. |
| `total_venta` | **Variable Objetivo (Target)**. |
| `medio_pago` | Forma de pago (Efectivo, Tarjeta, etc.). |
| `promo_producto` / `promo_pago` | Promociones aplicadas. |

## ⚙️ Metodología

### 1. Data Wrangling (Limpieza)
- **Detección de duplicados:** Eliminación de registros redundantes.
- **Manejo de Nulos:** Imputación en columnas de promociones y limpieza de registros críticos incompletos.
- **Outliers:** Análisis de valores atípicos mediante el Rango Intercuartil (IQR) en precios y cantidades.
- **Transformación:** Conversión de fechas y estandarización de columnas.

### 2. Análisis Exploratorio (EDA)
Se realizaron análisis univariados, bivariados y multivariados para validar tres hipótesis principales sobre el comportamiento de las ventas.

### 3. Machine Learning (Predicción)
- **Preprocesamiento:** Encoding de variables categóricas (LabelEncoder).
- **Feature Selection:** Selección de las 5 variables más relevantes usando `SelectKBest` (f_regression).
- **Modelo:** Regresión Lineal (`LinearRegression`).
- **Validación:** División Train/Test (80/20) y cálculo de métricas de error.

## 💡 Hallazgos del EDA
Durante el análisis se validaron las siguientes hipótesis:

1.  **Dominio de Categorías:** Se confirmó que las "Herramientas eléctricas y manuales" representan la mayor parte del volumen de ventas (**Hipótesis 1 Confirmada**).
2.  **Comportamiento de Pago:** Contrario a lo esperado, el **Efectivo** registró un Ticket Promedio superior al de las Tarjetas de Crédito, impulsado fuertemente por promociones de descuento (**Hipótesis 2 Refutada**).
3.  **Eficiencia por Sucursal:** Aunque Mendoza y Córdoba tienen más volumen, la sucursal de **Rosario** demostró tener el ticket promedio más alto, indicando una mayor eficiencia por cliente (**Hipótesis 3 Refutada**).

## 🤖 Modelo de Machine Learning

Se entrenó un modelo de regresión para predecir la variable `Total Venta`.

- **Algoritmo:** Regresión Lineal.
- **Features Seleccionadas:** El algoritmo seleccionó automáticamente variables clave como `Cantidad`, `Categoría` y `Sucursal` como los mejores predictores.
- **Métricas de Desempeño:**
    - **R² (R-Cuadrado):** Indica qué tan bien el modelo explica la varianza de los datos.
    - **MAE (Error Absoluto Medio):** Promedio de error en pesos ($) por predicción.
    - **RMSE:** Penalización de errores grandes.

*El modelo demostró ser capaz de identificar tendencias lineales, aunque se sugiere explorar modelos no lineales (como Random Forest) para capturar mejor la complejidad de los precios.*

## 💻 Requisitos y Ejecución

Para correr este proyecto localmente o en Google Colab, necesitarás las siguientes librerías:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
