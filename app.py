import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------
st.set_page_config(page_title="Housing Price Prediction Dashboard", layout="wide")

st.title("Housing Price Prediction Dashboard")
st.subheader("Proyecto de Análisis de Negocios y Soluciones")

# -----------------------------
# CARGA DE DATOS
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

# -----------------------------
# PREPARACIÓN DE DATOS
# -----------------------------
features = [
    'median_income',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households'
]

target = 'median_house_value'

X = df[features]
y = df[target]

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODELO
# -----------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

rf_model = train_model(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# -----------------------------
# MÉTRICAS
# -----------------------------
mae = mean_absolute_error(y_test, rf_pred)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
r2 = r2_score(y_test, rf_pred)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navegación")
section = st.sidebar.radio(
    "Ir a:",
    [
        "Resumen del negocio",
        "Exploración de datos",
        "Modelo y métricas",
        "Predicción interactiva",
        "Importancia de variables"
    ]
)

# -----------------------------
# 1. RESUMEN DEL NEGOCIO
# -----------------------------
if section == "Resumen del negocio":
    st.header("Resumen del problema de negocio")

    st.write("""
    El mercado inmobiliario presenta alta variabilidad en los precios de las viviendas, lo que dificulta
    para compradores, vendedores e inversionistas estimar el valor adecuado de una propiedad.

    Este proyecto utiliza datos históricos de vivienda para identificar variables relevantes y construir
    un modelo predictivo que ayude a estimar el valor de una vivienda con base en sus características.
    """)

    st.write("### Objetivo de negocio")
    st.write("""
    Apoyar decisiones relacionadas con:
    - fijación de precios
    - comparación entre propiedades
    - identificación de factores que influyen en el valor de mercado
    """)

    st.write("### Dataset utilizado")
    st.write("""
    Se utilizó un dataset de viviendas con variables como:
    - ingreso medio del área
    - antigüedad de la vivienda
    - número de habitaciones
    - número de dormitorios
    - población
    - número de hogares
    """)

# -----------------------------
# 2. EXPLORACIÓN DE DATOS
# -----------------------------
elif section == "Exploración de datos":
    st.header("Exploración de datos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución del precio de las viviendas")
        fig, ax = plt.subplots()
        ax.hist(df['median_house_value'], bins=40)
        ax.set_title("Distribución del precio")
        ax.set_xlabel("Precio")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    with col2:
        st.subheader("Ingreso medio vs valor de la vivienda")
        fig, ax = plt.subplots()
        ax.scatter(df['median_income'], df['median_house_value'], alpha=0.3)
        ax.set_title("Ingreso medio vs precio")
        ax.set_xlabel("Ingreso medio")
        ax.set_ylabel("Valor de la vivienda")
        st.pyplot(fig)

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(20))

# -----------------------------
# 3. MODELO Y MÉTRICAS
# -----------------------------
elif section == "Modelo y métricas":
    st.header("Modelo y métricas")

    st.write("Se entrenó un modelo **Random Forest Regressor** para estimar el valor de las viviendas.")

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("R²", f"{r2:.4f}")

    st.subheader("Valores reales vs predichos")
    fig, ax = plt.subplots()
    ax.scatter(y_test, rf_pred, alpha=0.4)
    ax.set_title("Random Forest: valores reales vs predichos")
    ax.set_xlabel("Valores reales")
    ax.set_ylabel("Valores predichos")
    st.pyplot(fig)

# -----------------------------
# 4. PREDICCIÓN INTERACTIVA
# -----------------------------
elif section == "Predicción interactiva":
    st.header("Predicción interactiva del precio de una vivienda")

    st.write("Ajusta los valores para estimar el precio de una vivienda:")

    median_income = st.slider("Ingreso medio del área", 0.0, 15.0, 5.0, 0.1)
    housing_median_age = st.slider("Antigüedad de la vivienda", 1, 60, 20)
    total_rooms = st.slider("Total de habitaciones", 1, 10000, 2000)
    total_bedrooms = st.slider("Total de dormitorios", 1, 3000, 400)
    population = st.slider("Población del área", 1, 20000, 1000)
    households = st.slider("Número de hogares", 1, 5000, 300)

    input_data = pd.DataFrame({
        'median_income': [median_income],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households]
    })

    prediction = rf_model.predict(input_data)[0]

    st.subheader("Precio estimado")
    st.success(f"${prediction:,.2f}")

    st.write("### Datos usados para la predicción")
    st.dataframe(input_data)

# -----------------------------
# 5. IMPORTANCIA DE VARIABLES
# -----------------------------
elif section == "Importancia de variables":
    st.header("Importancia de variables")

    importancias = pd.DataFrame({
        'Variable': features,
        'Importancia': rf_model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    st.subheader("Tabla de importancia")
    st.dataframe(importancias)

    st.subheader("Gráfica de importancia")
    fig, ax = plt.subplots()
    ax.bar(importancias['Variable'], importancias['Importancia'])
    ax.set_title("Importancia de variables - Random Forest")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Importancia")
    plt.xticks(rotation=45)
    st.pyplot(fig)
