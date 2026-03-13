import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------
# CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Housing Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# DATA
# ---------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

features = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households"
]
target = "median_house_value"

X = df[features].copy()
y = df[target].copy()

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# MODEL
# ---------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

rf_model = train_model(X_train, y_train)
rf_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, rf_pred)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
r2 = r2_score(y_test, rf_pred)

importancias = pd.DataFrame({
    "Variable": features,
    "Importancia": rf_model.feature_importances_
}).sort_values(by="Importancia", ascending=False)

pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": rf_pred
})

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.title("🏠 Navegación")
section = st.sidebar.radio(
    "Ir a:",
    [
        "Overview",
        "Exploración interactiva",
        "Modelo predictivo",
        "Simulador de precio"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros para exploración")

income_range = st.sidebar.slider(
    "Ingreso medio",
    float(df["median_income"].min()),
    float(df["median_income"].max()),
    (
        float(df["median_income"].min()),
        float(df["median_income"].max())
    )
)

age_range = st.sidebar.slider(
    "Antigüedad de la vivienda",
    int(df["housing_median_age"].min()),
    int(df["housing_median_age"].max()),
    (
        int(df["housing_median_age"].min()),
        int(df["housing_median_age"].max())
    )
)

filtered_df = df[
    (df["median_income"] >= income_range[0]) &
    (df["median_income"] <= income_range[1]) &
    (df["housing_median_age"] >= age_range[0]) &
    (df["housing_median_age"] <= age_range[1])
]

# ---------------------------------
# HEADER
# ---------------------------------
st.title("Housing Analytics & Price Prediction")
st.caption("Proyecto de Análisis de Negocios y Soluciones")

# ---------------------------------
# OVERVIEW
# ---------------------------------
if section == "Overview":
    st.markdown("## Resumen ejecutivo")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros analizados", f"{len(df):,}")
    col2.metric("Precio promedio", f"${df[target].mean():,.0f}")
    col3.metric("RMSE del modelo", f"{rmse:,.0f}")
    col4.metric("R² del modelo", f"{r2:.3f}")

    st.markdown("### Problema de negocio")
    st.write("""
    Estimar el precio de una vivienda puede ser complejo porque depende de varias variables al mismo tiempo.
    Este proyecto busca apoyar decisiones de negocio relacionadas con pricing, comparación de propiedades
    y evaluación de oportunidades inmobiliarias.
    """)

    st.markdown("### Hallazgos clave")
    top_var = importancias.iloc[0]["Variable"]
    st.write(f"""
    - La variable con mayor peso en la predicción es **{top_var}**.
    - Existe una relación positiva clara entre **median_income** y el valor de la vivienda.
    - El modelo Random Forest ofrece una base útil para crear una herramienta de estimación interactiva.
    """)

    fig_hist = px.histogram(
        filtered_df,
        x="median_house_value",
        nbins=40,
        title="Distribución del precio de las viviendas"
    )
    fig_hist.update_layout(height=450)

    fig_scatter = px.scatter(
        filtered_df,
        x="median_income",
        y="median_house_value",
        opacity=0.5,
        title="Ingreso medio vs valor de la vivienda",
        hover_data=features
    )
    fig_scatter.update_layout(height=450)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------
# EXPLORACIÓN
# ---------------------------------
elif section == "Exploración interactiva":
    st.markdown("## Exploración interactiva")

    tab1, tab2, tab3 = st.tabs(["Relaciones", "Correlación", "Datos"])

    with tab1:
        x_axis = st.selectbox(
            "Selecciona variable para eje X",
            options=features,
            index=0
        )

        y_axis = st.selectbox(
            "Selecciona variable para eje Y",
            options=["median_house_value"] + features,
            index=0
        )

        size_option = st.selectbox(
            "Tamaño de puntos",
            options=[None] + features,
            index=0
        )

        fig_dynamic = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            size=size_option if size_option is not None else None,
            color="median_house_value",
            hover_data=features,
            title=f"{x_axis} vs {y_axis}"
        )
        fig_dynamic.update_layout(height=600)
        st.plotly_chart(fig_dynamic, use_container_width=True)

    with tab2:
        corr = filtered_df[features + [target]].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Mapa de correlación"
        )
        fig_corr.update_layout(height=650)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.dataframe(filtered_df, use_container_width=True)

# ---------------------------------
# MODELO
# ---------------------------------
elif section == "Modelo predictivo":
    st.markdown("## Modelo predictivo")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.0f}")
    c2.metric("RMSE", f"{rmse:,.0f}")
    c3.metric("R²", f"{r2:.3f}")

    tab1, tab2 = st.tabs(["Actual vs Predicted", "Importancia de variables"])

    with tab1:
        fig_pred = px.scatter(
            pred_df,
            x="Actual",
            y="Predicted",
            opacity=0.5,
            title="Valores reales vs valores predichos"
        )

        min_val = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
        max_val = max(pred_df["Actual"].max(), pred_df["Predicted"].max())

        fig_pred.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(dash="dash")
        )

        fig_pred.update_layout(height=600)
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        fig_imp = px.bar(
            importancias,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Importancia de variables en Random Forest"
        )
        fig_imp.update_layout(height=500, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_imp, use_container_width=True)
        st.dataframe(importancias, use_container_width=True)

# ---------------------------------
# SIMULADOR
# ---------------------------------
elif section == "Simulador de precio":
    st.markdown("## Simulador interactivo de precio")

    st.write("Ajusta los valores y explora cómo cambia la predicción del modelo.")

    c1, c2 = st.columns(2)

    with c1:
        median_income = st.slider("Ingreso medio del área", 0.0, 15.0, 5.0, 0.1)
        housing_median_age = st.slider("Antigüedad de la vivienda", 1, 60, 20)
        total_rooms = st.slider("Total de habitaciones", 1, 10000, 2000)

    with c2:
        total_bedrooms = st.slider("Total de dormitorios", 1, 3000, 400)
        population = st.slider("Población del área", 1, 20000, 1000)
        households = st.slider("Número de hogares", 1, 5000, 300)

    input_data = pd.DataFrame({
        "median_income": [median_income],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households]
    })

    prediction = rf_model.predict(input_data)[0]

    st.metric("Precio estimado de la vivienda", f"${prediction:,.0f}")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={"text": "Predicción del modelo"},
        gauge={
            "axis": {"range": [0, float(df[target].max())]}
        }
    ))
    fig_gauge.update_layout(height=400)

    c3, c4 = st.columns([1.2, 1])
    with c3:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with c4:
        st.dataframe(input_data, use_container_width=True)
