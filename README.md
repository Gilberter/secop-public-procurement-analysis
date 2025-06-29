# secop-public-procurement-analysis

# 📊 Análisis y Predicción de Contratos Públicos en Colombia (SECOP)

Este proyecto analiza datos del Sistema Electrónico de Contratación Pública (SECOP) en Colombia, utilizando técnicas de procesamiento de lenguaje natural (NLP) para extraer información clave de descripciones contractuales, y modelos de regresión para predecir el valor de los contratos. Además, se exploró la capacidad de Amazon SageMaker para escalar este proceso mediante entrenamiento en la nube.

---

## 🚀 Objetivos

- Extraer entidades y frases clave de descripciones de contratos públicos usando Amazon Comprehend.
- Analizar la frecuencia y tipo de estas entidades y frases para obtener insights sobre el comportamiento contractual.
- Entrenar un modelo de regresión para predecir el `valor_contrato` a partir de las entidades detectadas.
- Implementar el entrenamiento del modelo en **Amazon SageMaker** para aprovechar su capacidad de cómputo escalable.

---

## 🧠 Tecnologías Utilizadas

| Herramienta / Servicio         | Propósito                                                                 |
|-------------------------------|--------------------------------------------------------------------------|
| **Python (Pandas, NumPy)**     | Limpieza, transformación y análisis de datos                            |
| **Scikit-learn**              | Modelado predictivo (regresión) y vectorización de texto (`CountVectorizer`) |
| **Amazon Comprehend**         | Extracción automática de entidades y frases clave desde texto (NLP)     |
| **Amazon SageMaker**          | Entrenamiento y gestión de modelos de machine learning en la nube       |
| **Amazon S3**                 | Almacenamiento de datos de entrada y salida del modelo                  |
| **Amazon IAM**                | Control de acceso seguro para los servicios SageMaker y Comprehend     |
| **Matplotlib / Seaborn**      | Visualización de datos exploratorios                                    |
| **Jupyter / SageMaker Studio**| Desarrollo y ejecución de notebooks en la nube (opcional)              |
| **Joblib**                    | Serialización del modelo entrenado                                      |

---

## 📈 Resultados

- Se realizó un análisis exploratorio visual (EDA) para identificar las frases clave y entidades más frecuentes en el dataset.
- Se entrenó un modelo de regresión lineal utilizando como entrada las entidades extraídas de cada contrato.
- El pipeline fue implementado localmente y adaptado para su ejecución remota en **Amazon SageMaker** mediante `SKLearn Estimator`.

---

## ⚠️ Reflexiones sobre el Entrenamiento

Aunque el modelo fue entrenado exitosamente en **Amazon SageMaker**, los resultados predictivos no fueron satisfactorios. Algunas de las razones clave:

- El valor de los contratos presenta una alta variabilidad y depende de muchos factores contextuales que no están presentes solo en el texto.
- Las entidades extraídas son útiles, pero no suficientes por sí solas como variables predictoras.
- Sería recomendable incorporar variables adicionales como la fecha, entidad contratante, modalidad de selección y tipo de contrato.
