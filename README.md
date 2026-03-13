# Housing Price Analysis and Prediction

Este repositorio contiene el desarrollo de un proyecto de **Análisis de Negocios y Soluciones** enfocado en analizar patrones dentro del mercado inmobiliario y desarrollar un modelo predictivo para estimar el valor de viviendas utilizando datos socioeconómicos y características del entorno.

El proyecto combina **análisis exploratorio de datos, modelado predictivo y visualización interactiva**, con el objetivo de transformar los resultados del análisis en una herramienta que permita explorar los datos y generar estimaciones de precios de forma dinámica.

---

# Objetivo del proyecto

El valor de las viviendas puede variar significativamente dependiendo de diferentes factores como el ingreso promedio del área, la ubicación geográfica, la densidad poblacional y características de las viviendas.

El objetivo de este proyecto es:

- Analizar los factores que influyen en el valor de las viviendas
- Identificar patrones dentro de los datos
- Desarrollar un modelo predictivo para estimar precios
- Crear un dashboard interactivo para explorar los resultados

Este tipo de análisis puede apoyar procesos de **análisis de mercado inmobiliario y toma de decisiones basada en datos**.

---

# Dataset

Para este proyecto se utilizó el **California Housing Dataset**, que contiene información demográfica y características de viviendas de diferentes distritos censales en California.

El dataset incluye variables como:

- ingreso medio del área
- antigüedad de las viviendas
- número total de habitaciones
- número total de dormitorios
- población del área
- número de hogares
- proximidad al océano

La variable objetivo del análisis es:

**median_house_value**, que representa el valor mediano de las viviendas.

---

# Metodología

El proyecto sigue una estructura similar al proceso **CRISP-DM** utilizado en proyectos de ciencia de datos:

1. **Business Understanding**  
   Definición del problema y del objetivo del análisis.

2. **Data Understanding**  
   Exploración inicial del dataset para comprender las variables disponibles.

3. **Exploratory Data Analysis**  
   Identificación de patrones, tendencias y relaciones entre variables.

4. **Data Preparation**  
   Limpieza de datos, manejo de valores faltantes y selección de variables.

5. **Modeling**  
   Construcción de un modelo predictivo utilizando Random Forest Regressor.

6. **Evaluation**  
   Evaluación del desempeño del modelo mediante métricas de regresión.

7. **Business Proposal**  
   Interpretación de resultados y posibles aplicaciones del análisis.

---

# Modelo Predictivo

Para estimar el valor de las viviendas se utilizó un modelo de **Random Forest Regressor**.

Este modelo fue seleccionado debido a su capacidad para:

- capturar relaciones no lineales entre variables
- manejar múltiples variables predictoras
- identificar la importancia relativa de cada variable

El modelo fue evaluado utilizando las siguientes métricas:

**MAE (Mean Absolute Error)**  
Mide el error promedio entre los valores reales y los predichos.

**RMSE (Root Mean Squared Error)**  
Penaliza errores grandes y proporciona una medida del error general del modelo.

**R² (Coeficiente de determinación)**  
Indica qué proporción de la variabilidad del precio de las viviendas puede ser explicada por el modelo.

---

# Dashboard interactivo

Para complementar el análisis se desarrolló un **dashboard interactivo utilizando Streamlit**.

El dashboard permite:

- explorar visualizaciones interactivas
- analizar precios por zona
- visualizar la distribución geográfica de las viviendas
- observar el desempeño del modelo predictivo
- utilizar un simulador interactivo para estimar el precio de una vivienda

Esto permite transformar el análisis en una herramienta que facilita la exploración de los datos.

---

# Estructura del repositorio
