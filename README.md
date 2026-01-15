# 🔍 Detección de Fraude en Tarjetas de Crédito

Sistema avanzado de detección de fraude en transacciones con tarjetas de crédito utilizando Machine Learning. Este proyecto implementa un modelo XGBoost que logra una precisión del **99.94%** en la identificación de transacciones fraudulentas.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características Principales](#-características-principales)
- [Métricas del Modelo](#-métricas-del-modelo)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
  - [Entrenamiento del Modelo](#entrenamiento-del-modelo)
  - [Uso del Modelo Entrenado](#uso-del-modelo-entrenado)
- [Análisis Exploratorio de Datos (EDA)](#-análisis-exploratorio-de-datos-eda)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Resultados y Visualizaciones](#-resultados-y-visualizaciones)
- [Características Más Importantes](#-características-más-importantes)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

## 🎯 Descripción

Este proyecto implementa un sistema completo de detección de fraude que incluye:

- **Análisis Exploratorio de Datos (EDA)** exhaustivo con visualizaciones
- **Preprocesamiento avanzado** de datos con escalado robusto
- **Modelo XGBoost** optimizado para detección de fraude
- **Evaluación completa** con múltiples métricas y validación cruzada
- **Sistema de predicción** listo para usar en producción
- **Visualizaciones profesionales** de resultados y análisis

El dataset utilizado contiene más de **568,000 transacciones** con características anonimizadas (V1-V28) y el monto de la transacción, clasificadas como normales o fraudulentas.

## ✨ Características Principales

- ✅ **Alta Precisión**: Modelo con 99.94% de accuracy y 100% de recall
- ✅ **EDA Completo**: Análisis exploratorio con 8 visualizaciones profesionales
- ✅ **Preprocesamiento Robusto**: Manejo de outliers y escalado adecuado
- ✅ **Validación Cruzada**: 5-fold cross-validation para garantizar robustez
- ✅ **Sistema de Predicción**: Función lista para usar con datos individuales o en lote
- ✅ **Visualizaciones**: Gráficos de alta calidad guardados automáticamente
- ✅ **Documentación**: Código bien documentado y notebooks explicativos

## 📊 Métricas del Modelo

El modelo XGBoost entrenado alcanza las siguientes métricas en el conjunto de prueba:

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 99.94% |
| **Precision (Fraude)** | 99.89% |
| **Recall (Sensibilidad)** | 100.00% |
| **Specificity** | 99.89% |
| **F1-Score** | 99.94% |
| **AUC-ROC** | 99.998% |
| **AUC-PR** | 99.998% |

### Validación Cruzada (5-fold)

- **AUC-PR**: 1.0000 (±0.0000)
- **AUC-ROC**: 1.0000 (±0.0000)
- **F1-Score**: 0.9995 (±0.0003)

## 📁 Estructura del Proyecto

```
Deteccion_Fraude_Credit_Card/
│
├── Archivos_CSV/
│   ├── creditcard_2023.csv              # Dataset principal
│   ├── importancia_features.csv          # Importancia de características
│   └── resultados_modelo.csv             # Métricas del modelo
│
├── Modelo_Entrenado/
│   ├── modelo_xgboost_fraude.pkl        # Modelo entrenado (XGBoost)
│   ├── scaler_robust.pkl                # Scaler para preprocesamiento
│   ├── features_names.pkl               # Nombres de características
│   └── metadata_modelo.txt              # Metadatos del modelo
│
├── graficos_eda/
│   ├── 01_distribucion_clases.png
│   ├── 02_distribucion_amount.png
│   ├── 03_correlacion_top_features.png
│   ├── 04_distribucion_top_features.png
│   ├── 05_matriz_correlacion.png
│   ├── 06_importancia_features.png
│   ├── 07_matriz_confusion.png
│   └── 08_curvas_roc_pr.png
│
├── fraude_credit_card.ipynb             # Notebook principal (EDA + Entrenamiento)
├── usar_modelo_entrenado.ipynb          # Notebook para usar el modelo
└── README.md                            # Este archivo
```

## 🔧 Requisitos

El proyecto requiere las siguientes librerías de Python:

- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`
- `scikit-learn >= 0.24.0`
- `xgboost >= 1.5.0`
- `imbalanced-learn >= 0.8.0`
- `joblib >= 1.0.0`

## 📦 Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/tu-usuario/Deteccion_Fraude_Credit_Card.git
cd Deteccion_Fraude_Credit_Card
```

2. **Crear un entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar las dependencias**:
```bash
pip install -r requirements.txt
```

Si no tienes un archivo `requirements.txt`, instala las dependencias manualmente:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
```

## 🚀 Uso

### Entrenamiento del Modelo

Para entrenar el modelo desde cero:

1. Asegúrate de tener el dataset `creditcard_2023.csv` en la carpeta `Archivos_CSV/`
2. Abre y ejecuta el notebook `fraude_credit_card.ipynb`
3. El notebook realizará automáticamente:
   - Análisis exploratorio de datos
   - Preprocesamiento
   - Entrenamiento del modelo XGBoost
   - Evaluación y generación de métricas
   - Guardado del modelo y visualizaciones

**Nota**: El entrenamiento puede tomar varios minutos dependiendo de tu hardware.

### Uso del Modelo Entrenado

Para usar el modelo ya entrenado en nuevas predicciones:

1. Abre el notebook `usar_modelo_entrenado.ipynb`
2. El notebook incluye ejemplos de:
   - **Predicción individual**: Predice si una transacción específica es fraudulenta
   - **Predicción en lote**: Procesa múltiples transacciones a la vez
   - **Predicción desde CSV**: Carga transacciones desde un archivo CSV

#### Ejemplo de Uso - Predicción Individual

Abre el notebook `usar_modelo_entrenado.ipynb` y ejecuta las celdas. El notebook incluye una función `predecir_fraude()` que puedes usar así:

```python
# Ejecutar en el notebook usar_modelo_entrenado.ipynb
# La función predecir_fraude() está definida en el notebook

# Transacción de ejemplo
transaccion = {
    'V1': -1.359807134,
    'V2': -0.072781173,
    'V3': 2.536346738,
    'V4': 1.378155224,
    'V5': -0.33826177,
    'V6': 0.46238804,
    'V7': 0.239598554,
    'V8': 0.098697901,
    'V9': 0.36378697,
    'V10': 0.090794172,
    'V11': -0.55159953,
    'V12': -0.617800856,
    'V13': -0.991389847,
    'V14': -2.261873095,
    'V15': 0.524979725,
    'V16': 0.247998153,
    'V17': 0.771679402,
    'V18': 0.909412262,
    'V19': -0.68928096,
    'V20': -0.327641834,
    'V21': -0.139096572,
    'V22': -0.055352794,
    'V23': -0.059751841,
    'V24': 0.342207708,
    'V25': 0.389796345,
    'V26': 0.005857858,
    'V27': -0.013406374,
    'V28': -0.017969444,
    'Amount': 149.62
}

# Realizar predicción (ejecutar después de cargar el modelo en el notebook)
resultado = predecir_fraude(transaccion)
print(resultado)
```

#### Ejemplo de Uso - Predicción en Lote desde CSV

```python
# Ejecutar en el notebook usar_modelo_entrenado.ipynb
# Descomenta la sección "5. Uso con Archivo CSV" en el notebook

# El notebook incluye código para:
# 1. Cargar transacciones desde CSV
datos = pd.read_csv('nuevas_transacciones.csv')

# 2. Realizar predicciones
resultados = predecir_fraude(datos, mostrar_detalles=False)

# 3. Guardar resultados
resultados.to_csv('predicciones_resultados.csv', index=False)
```

## 📈 Análisis Exploratorio de Datos (EDA)

El proyecto incluye un EDA completo con las siguientes visualizaciones:

1. **Distribución de Clases**: Análisis del balance de clases (Normal vs Fraude)
2. **Distribución de Amount**: Análisis del monto de las transacciones por clase
3. **Correlación Top Features**: Top 10 características más correlacionadas con fraude
4. **Distribución de Features**: Distribuciones de las características más importantes
5. **Matriz de Correlación**: Correlación entre las 15 características más relevantes
6. **Importancia de Features**: Importancia de características según XGBoost
7. **Matriz de Confusión**: Visualización del rendimiento del modelo
8. **Curvas ROC y Precision-Recall**: Curvas de evaluación del modelo

Todas las visualizaciones se guardan automáticamente en la carpeta `graficos_eda/` en formato PNG de alta resolución (300 DPI).

## 🛠 Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje de programación principal
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas
- **Matplotlib & Seaborn**: Visualización de datos
- **Scikit-learn**: Preprocesamiento y métricas de evaluación
- **XGBoost**: Algoritmo de Machine Learning
- **Imbalanced-learn**: Manejo de clases desbalanceadas (SMOTE)
- **Joblib**: Serialización del modelo

## 📊 Resultados y Visualizaciones

### Características del Dataset

- **Total de transacciones**: 568,630
- **Características**: 29 (V1-V28 + Amount)
- **Clases**: Balanceadas (50% Normal, 50% Fraude)
- **Valores faltantes**: Ninguno

### Preprocesamiento

- Escalado con **RobustScaler** (resistente a outliers)
- División train-test estratificada (80/20)
- Validación de integridad de datos

### Parámetros del Modelo XGBoost

- **Objetivo**: `binary:logistic`
- **Métrica de evaluación**: `aucpr` (Area Under Precision-Recall Curve)
- **max_depth**: 6
- **learning_rate**: 0.1
- **n_estimators**: 200
- **subsample**: 0.8
- **colsample_bytree**: 0.8
- **min_child_weight**: 1
- **gamma**: 0.1

## 🎯 Características Más Importantes

Las características más importantes para la detección de fraude según el modelo son:

| Feature | Importancia | % del Total |
|---------|-------------|-------------|
| **V14** | 0.388 | 38.8% |
| **V10** | 0.250 | 25.0% |
| **V4** | 0.082 | 8.2% |
| **V17** | 0.032 | 3.2% |
| **V12** | 0.023 | 2.3% |

Estas 5 características representan aproximadamente el **77.5%** de la importancia total del modelo.

## 💡 Características del Sistema

- **Detección en Tiempo Real**: El modelo puede procesar transacciones individuales instantáneamente
- **Procesamiento en Lote**: Eficiente procesamiento de múltiples transacciones
- **Interpretabilidad**: Visualización de importancia de características
- **Robustez**: Validación cruzada garantiza buen rendimiento en datos nuevos
- **Escalabilidad**: Puede manejar grandes volúmenes de transacciones

## 📝 Notas Importantes

- El modelo utiliza un **umbral de decisión de 0.5** por defecto. Puede ajustarse según necesidades específicas (mayor sensibilidad vs. menor falsos positivos).
- Las características V1-V28 son resultado de una transformación PCA para proteger la privacidad, por lo que no son interpretables directamente.
- El modelo debe ser **re-entrenado periódicamente** con nuevos datos para mantener su efectividad.
- Se recomienda monitorear el rendimiento del modelo en producción y ajustar según sea necesario.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**Martin**

- GitHub: [@DataScienceWorld1805](https://github.com/DataScienceWorld1805)
- Email: datascienceworld1805@gmail.com

## 🙏 Agradecimientos

- Dataset: Credit Card Fraud Detection Dataset 2023
- Librerías open-source de la comunidad de Python y Machine Learning

---

⭐ Si te gustó este proyecto, ¡dale una estrella en GitHub!
