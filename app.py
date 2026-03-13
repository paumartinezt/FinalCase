import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------
st.set_page_config(
    page_title="Housing Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# ESTILO
# ---------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# CARGA DE DATOS
# ---------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

# ---------------------------------
# PREPARACIÓN DE DATOS
# ---------------------------------
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
# MODELO
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

# ---------------------------------
# MÉTRICAS
# ---------------------------------
mae = mean_absolute_error(y_test, rf_pred)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
r2 = r2_score(y_test, rf_pred)

# ---------------------------------
# DATAFRAMES AUXILIARES
# ---------------------------------
importancias = pd.DataFrame({
    "Variable": features,
    "Importancia": rf_model.feature_importances_
}).sort_values(by="Importancia", ascending=False)

pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": rf_pred
})

# Crear métrica visual más fácil de entender
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.title("🏠 Navegación")
section = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Overview",
        "Mapa interactivo",
        "Análisis por zona",
        "Modelo predictivo",
        "Simulador de precio"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

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

zone_options = sorted(df["ocean_proximity"].dropna().unique().tolist())
selected_zones = st.sidebar.multiselect(
    "Zona",
    options=zone_options,
    default=zone_options
)

filtered_df = df[
    (df["median_income"] >= income_range[0]) &
    (df["median_income"] <= income_range[1]) &
    (df["housing_median_age"] >= age_range[0]) &
    (df["housing_median_age"] <= age_range[1]) &
    (df["ocean_proximity"].isin(selected_zones))
].copy()

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown('<div class="main-title">Housing Analytics & Price Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Proyecto de Análisis de Negocios y Soluciones</div>',
    unsafe_allow_html=True
)

# ---------------------------------
# OVERVIEW
# ---------------------------------
if section == "Overview":
    st.markdown('<div class="section-title">Resumen ejecutivo</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros analizados", f"{len(filtered_df):,}")
    col2.metric("Precio promedio", f"${filtered_df[target].mean():,.0f}")
    col3.metric("RMSE del modelo", f"{rmse:,.0f}")
    col4.metric("R² del modelo", f"{r2:.3f}")

    st.markdown("### Problema de negocio")
    st.write("""
    Estimar el valor de una vivienda puede ser difícil porque depende de múltiples factores al mismo tiempo.
    Este proyecto busca apoyar decisiones de negocio relacionadas con fijación de precios, comparación de propiedades
    y análisis de oportunidades en el mercado inmobiliario.
    """)

    st.markdown("### Hallazgos clave")
    top_var = importancias.iloc[0]["Variable"]
    st.write(f"""
    - La variable más importante para el modelo es **{top_var}**.
    - Las viviendas en ciertas zonas muestran diferencias claras en precio promedio.
    - El modelo Random Forest permite crear una herramienta útil de estimación de precios.
    """)

    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            filtered_df,
            x="median_house_value",
            nbins=35,
            title="Distribución del precio de las viviendas",
            color_discrete_sequence=["#4F46E5"]
        )
        fig_hist.update_layout(
            xaxis_title="Precio de la vivienda",
            yaxis_title="Frecuencia",
            height=420
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        zona_df = (
            filtered_df.groupby("ocean_proximity", as_index=False)["median_house_value"]
            .mean()
            .sort_values("median_house_value", ascending=False)
        )

        fig_zone = px.bar(
            zona_df,
            x="ocean_proximity",
            y="median_house_value",
            text_auto=".0f",
            title="Precio promedio por zona",
            color="median_house_value",
            color_continuous_scale="Blues"
        )
        fig_zone.update_layout(
            xaxis_title="Zona",
            yaxis_title="Precio promedio",
            height=420,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_zone, use_container_width=True)

# ---------------------------------
# MAPA INTERACTIVO
# ---------------------------------
elif section == "Mapa interactivo":
    st.markdown('<div class="section-title">Mapa interactivo de precios</div>', unsafe_allow_html=True)

    st.write("""
    Este mapa permite explorar cómo cambia el valor de las viviendas según su ubicación.
    El color representa el precio y el tamaño del punto está relacionado con el ingreso medio del área.
    """)

    map_size_option = st.selectbox(
        "Tamaño de los puntos según:",
        options=["median_income", "population", "households"],
        index=0
    )

    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="median_house_value",
        size=map_size_option,
        hover_name="ocean_proximity",
        hover_data={
            "median_house_value": ":,.0f",
            "median_income": ":.2f",
            "housing_median_age": True,
            "total_rooms": True,
            "latitude": False,
            "longitude": False
        },
        zoom=4.5,
        height=650,
        title="Mapa del valor de las viviendas",
        color_continuous_scale="Viridis"
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------
# ANÁLISIS POR ZONA
# ---------------------------------
elif section == "Análisis por zona":
    st.markdown('<div class="section-title">Análisis visual por zona</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Precio por zona", "Composición", "Relación visual"])

    with tab1:
        zona_df = (
            filtered_df.groupby("ocean_proximity", as_index=False)
            .agg(
                precio_promedio=("median_house_value", "mean"),
                ingreso_promedio=("median_income", "mean"),
                total_viviendas=("median_house_value", "count")
            )
            .sort_values("precio_promedio", ascending=False)
        )

        fig_zone = px.bar(
            zona_df,
            x="ocean_proximity",
            y="precio_promedio",
            text_auto=".0f",
            color="ingreso_promedio",
            title="Precio promedio por zona",
            color_continuous_scale="Tealgrn"
        )
        fig_zone.update_layout(
            xaxis_title="Zona",
            yaxis_title="Precio promedio",
            height=500
        )
        st.plotly_chart(fig_zone, use_container_width=True)
        st.dataframe(zona_df, use_container_width=True)

    with tab2:
        count_df = (
            filtered_df["ocean_proximity"]
            .value_counts()
            .reset_index()
        )
        count_df.columns = ["Zona", "Cantidad"]

        fig_pie = px.pie(
            count_df,
            names="Zona",
            values="Cantidad",
            title="Distribución de viviendas por zona",
            hole=0.45
        )
        fig_pie.update_layout(height=500)
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        fig_bubble = px.scatter(
            filtered_df,
            x="rooms_per_household",
            y="median_house_value",
            size="median_income",
            color="ocean_proximity",
            hover_data=["housing_median_age", "population"],
            title="Habitaciones por hogar vs valor de la vivienda",
            opacity=0.65
        )
        fig_bubble.update_layout(
            xaxis_title="Habitaciones por hogar",
            yaxis_title="Valor de la vivienda",
            height=550
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

# ---------------------------------
# MODELO PREDICTIVO
# ---------------------------------
elif section == "Modelo predictivo":
    st.markdown('<div class="section-title">Desempeño del modelo predictivo</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.0f}")
    c2.metric("RMSE", f"{rmse:,.0f}")
    c3.metric("R²", f"{r2:.3f}")

    tab1, tab2 = st.tabs(["Real vs predicho", "Importancia de variables"])

    with tab1:
        fig_pred = px.scatter(
            pred_df,
            x="Actual",
            y="Predicted",
            opacity=0.5,
            title="Valores reales vs valores predichos",
            color="Predicted",
            color_continuous_scale="Blues"
        )

        min_val = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
        max_val = max(pred_df["Actual"].max(), pred_df["Predicted"].max())

        fig_pred.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(dash="dash", color="red")
        )

        fig_pred.update_layout(
            xaxis_title="Valor real",
            yaxis_title="Valor predicho",
            height=600
        )

        st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        fig_imp = px.bar(
            importancias,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Importancia de variables en Random Forest",
            color="Importancia",
            color_continuous_scale="Purples"
        )
        fig_imp.update_layout(
            height=500,
            yaxis=dict(categoryorder="total ascending")
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.dataframe(importancias, use_container_width=True)

# ---------------------------------
# SIMULADOR DE PRECIO
# ---------------------------------
elif section == "Simulador de precio":
    st.markdown('<div class="section-title">Simulador interactivo de precio</div>', unsafe_allow_html=True)

    st.write("Ajusta los valores y observa cómo cambia la predicción del modelo.")

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
    max_price = float(df["median_house_value"].max())
    fill_pct = min(prediction / max_price, 1.0)

    st.metric("Precio estimado de la vivienda", f"${prediction:,.0f}")

    c3, c4 = st.columns([1.2, 1])

    with c3:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "Predicción del modelo"},
            gauge={
                "axis": {"range": [0, max_price]},
                "bar": {"thickness": 0.35},
                "steps": [
                    {"range": [0, max_price * 0.33], "color": "#dbeafe"},
                    {"range": [max_price * 0.33, max_price * 0.66], "color": "#93c5fd"},
                    {"range": [max_price * 0.66, max_price], "color": "#2563eb"}
                ]
            }
        ))
        fig_gauge.update_layout(height=380)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c4:
        st.markdown("### Nivel visual del valor")
        st.markdown("## 🏠")
        st.progress(fill_pct)
        st.write(f"Esta estimación representa **{fill_pct:.0%}** del valor máximo observado en el dataset.")
        st.dataframe(input_data, use_container_width=True)
