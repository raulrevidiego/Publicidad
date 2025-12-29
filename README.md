# 📢 Publicidad – Análisis de Datos y Machine Learning

Este proyecto desarrolla un **modelo de Machine Learning supervisado** aplicado a un problema realista de **publicidad**, con el objetivo de **predecir si un usuario responderá positivamente a una campaña publicitaria** a partir de sus características demográficas y de comportamiento.

El trabajo cubre todo el flujo típico de un proyecto de Data Science:  
**EDA → preprocesamiento → entrenamiento → evaluación del modelo**.

---

## 🎯 Objetivo

Predecir la **respuesta del usuario ante una acción publicitaria** (por ejemplo, clic o compra) utilizando variables como edad, ingresos y hábitos de uso, y evaluar el rendimiento del modelo con métricas de clasificación.

Este tipo de problema es común en:
- Marketing digital
- Segmentación de usuarios
- Optimización de campañas publicitarias

---

## 📊 Dataset

- Tipo de problema: **Clasificación binaria**
- Variable objetivo:
  - Respuesta del usuario ante la publicidad (0 / 1)
- Variables utilizadas:
  - `Age` → Edad del usuario
  - `EstimatedSalary` → Ingresos estimados
  - Otras variables de comportamiento según el dataset

El dataset permite analizar cómo influyen factores demográficos y económicos en la probabilidad de respuesta a una campaña.

---

## 🔍 Análisis Exploratorio de Datos (EDA)

Durante el análisis exploratorio se estudian:

- Distribución de las variables numéricas
- Diferencias entre usuarios que responden y los que no
- Relación entre edad, salario y respuesta publicitaria
- Identificación de patrones y separabilidad entre clases

Se utilizan visualizaciones para comprobar si los datos contienen información suficiente para un modelo predictivo.

---

## 🧹 Preprocesamiento de Datos

Las tareas de preparación incluyen:

- Selección de variables relevantes
- Separación entre variables predictoras (`X`) y variable objetivo (`y`)
- División del dataset en:
  - Conjunto de entrenamiento
  - Conjunto de test
- Escalado de variables numéricas cuando es necesario

Se siguen buenas prácticas para evitar **data leakage**.

---

## 🤖 Machine Learning

Se entrena un modelo de **clasificación supervisada** siguiendo el flujo estándar:

1. Entrenamiento del modelo con los datos de entrenamiento  
2. Predicción sobre el conjunto de test  
3. Evaluación mediante métricas de clasificación  

### Métricas utilizadas:
- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión

El objetivo no es solo obtener una buena métrica, sino **entender el comportamiento del modelo y sus errores**.

---

## 🧠 Resultados y Conclusiones

A partir del análisis y del modelo entrenado se observa que:

- Edad e ingresos tienen un impacto claro en la probabilidad de respuesta
- El escalado de variables mejora el rendimiento del modelo
- El modelo es capaz de capturar patrones útiles para segmentación de usuarios
- Este enfoque puede servir como base para optimizar campañas publicitarias reales

---

## 🛠️ Tecnologías Utilizadas

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🚀 Posibles Mejoras Futuras

- Comparación entre distintos algoritmos de clasificación
- Ajuste de hiperparámetros
- Validación cruzada
- Ingeniería de características
- Evaluación con métricas orientadas a negocio
- Análisis de importancia de variables

---

## 👤 Autor

Proyecto realizado por **Raúl Revidiego**  
Enfocado en aprendizaje práctico y aplicación de Machine Learning a problemas reales de negocio.

---

## 📌 Nota

Proyecto con fines educativos y demostrativos dentro del ámbito de **Ciencia de Datos y Machine Learning**.

