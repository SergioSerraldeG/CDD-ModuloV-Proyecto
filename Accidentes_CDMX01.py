import streamlit as st
import pandas as pd
import pickle
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# -------------------------------------------
# CARGA DE DATOS Y MODELO
# -------------------------------------------

@st.cache_data
def cargar_df():
    return pd.read_csv("/Users/mariafernandaserraldegarces/Downloads/Accidentes - Modulo V/df.csv")

@st.cache_resource
def cargar_modelo():
    model_obj = pickle.load(open("/Users/mariafernandaserraldegarces/Downloads/Accidentes - Modulo V/modelo_xgb_full.pkl", "rb"))
    
    # Si el pickle trae dict ─ acceder al XGBClassifier
    if isinstance(model_obj, dict):
        if "model" in model_obj:
            return model_obj["model"]
        else:
            st.error("No se encontró 'model' dentro del pickle.")
            st.stop()
    else:
        return model_obj

df = cargar_df()
model = cargar_modelo()


# -------------------------------------------
# FUNCIÓN PARA GENERAR PREDICCIONES
# -------------------------------------------

def generar_predicciones(df):
    df_temp = df.copy()
    
    # Variables para el modelo → quitamos lat/long
    X = df_temp.drop(columns=["latitud", "longitud"], errors="ignore")

    # Predicción
    proba = model.predict_proba(X)[:, 1]

    # Agregar columnas
    df_temp["riesgo"] = proba
    df_temp["color"] = df_temp["riesgo"].apply(
        lambda x: "red" if x >= 0.7 else ("orange" if x >= 0.4 else "green")
    )

    return df_temp


df_pred = generar_predicciones(df)


# -------------------------------------------
# CONFIGURACIÓN STREAMLIT
# -------------------------------------------

st.set_page_config(layout="wide")
st.title("🛣️ Plataforma de Riesgo de Transporte CDMX – Versión Unificada")


# -------------------------------------------
# MAPA CALENTADO + PUNTOS
# -------------------------------------------

st.subheader("Mapa de calor + puntos por predicción del modelo")

center_lat = df["latitud"].mean()
center_lon = df["longitud"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# Heatmap basado en riesgo del modelo
heat_data = df_pred[["latitud", "longitud", "riesgo"]].values.tolist()
HeatMap(heat_data, radius=13, blur=15).add_to(m)

# Puntos coloreados por riesgo
for _, row in df_pred.iterrows():
    folium.CircleMarker(
        location=[row["latitud"], row["longitud"]],
        radius=4,
        color=row["color"],
        fill=True,
        fill_color=row["color"],
        fill_opacity=0.8,
        popup=f"Riesgo: {row['riesgo']:.2f}"
    ).add_to(m)

st_folium(m, width=900, height=600)


# -------------------------------------------
# BÚSQUEDA POR DIRECCIÓN – COMO EN V1
# -------------------------------------------

st.subheader("Buscar dirección y ver riesgo")

direccion = st.text_input("Escribe una dirección en CDMX (ejemplo: Reforma 222)")

if direccion:
    try:
        geolocator = Nominatim(user_agent="riesgo_cdmx")
        location = geolocator.geocode("CDMX " + direccion)

        if location:
            lat = location.latitude
            lon = location.longitude

            st.success(f"📍 Dirección encontrada: {location.address}")
            st.write(f"Latitud: {lat}, Longitud: {lon}")

            # Crear df con una sola fila
            df_single = df_pred.iloc[[0]].copy()
            df_single["latitud"] = lat
            df_single["longitud"] = lon

            # Predecir para este punto
            punto_pred = generar_predicciones(df_single).iloc[0]

            st.write(f"Riesgo en esta ubicación: **{punto_pred['riesgo']:.2f}**")

            # Mapa de un solo punto
            m2 = folium.Map(location=[lat, lon], zoom_start=15)
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                color=punto_pred["color"],
                fill=True,
                fill_color=punto_pred["color"],
                popup=f"{direccion} – Riesgo {punto_pred['riesgo']:.2f}"
            ).add_to(m2)

            st_folium(m2, width=900, height=500)

        else:
            st.error("No se encontró la dirección.")

    except Exception as e:
        st.error(f"Ocurrió un error geocodificando: {e}")

