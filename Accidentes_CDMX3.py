import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import requests
import datetime
import json
import pandas as pd
import joblib
import numpy as np
import pydeck as pdk
from sklearn.preprocessing import LabelEncoder

# =============================
#   CARGA DE VARIABLES Y MODELO
# =============================

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OWM_API = os.getenv("OWM_API_KEY")
TOMTOM_API = os.getenv("TOMTOM_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
model_openai = "gpt-4o-mini"

# Cargar modelo
modelo_data = joblib.load("/Users/mariafernandaserraldegarces/Downloads/Accidentes - Modulo V/modelo_xgb_full.pkl")
modelo_xgb = modelo_data["model"]
modelo_features = modelo_data["features"]

# IMPORTANTE: Cargar LabelEncoder para alcaldías
le = LabelEncoder()
alcaldias_ref = [
    "Álvaro Obregón", "Azcapotzalco", "Benito Juárez", "Coyoacán", "Cuajimalpa",
    "Cuauhtémoc", "Gustavo A. Madero", "Iztacalco", "Iztapalapa",
    "La Magdalena Contreras", "Miguel Hidalgo", "Milpa Alta", "Tláhuac",
    "Tlalpan", "Venustiano Carranza", "Xochimilco"
]
le.fit(alcaldias_ref)


# =============================
#  FUNCIONES AUXILIARES
# =============================

def obtener_alcaldia_desde_api(lat, lon):
    """Obtiene alcaldía desde coordenadas usando TomTom reverse geocoding."""
    url = f"https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json?key={TOMTOM_API}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    address = data["addresses"][0]["address"]

    if "municipality" in address:
        return address["municipality"]
    return "Desconocido"


def predecir_riesgo_modelo(alcaldia, clima, temp, tmin, tmax, prcp, wind, hora):
    """Construye variables del modelo y predice probabilidad con XGBoost."""

    fin_semana = 1 if datetime.datetime.now().weekday() >= 5 else 0
    hora_pico = 1 if hora in [6, 7, 8, 17, 18, 19] else 0

    lluvia_bin = 1 if "rain" in clima else 0
    lluvia_fuerte = 1 if "heavy" in clima or prcp > 8 else 0
    riesgo_frio = 1 if temp < 10 else 0
    pav_humedo_frio = 1 if lluvia_bin == 1 and temp < 15 else 0

    df_pred = pd.DataFrame([{
        "alcaldia": alcaldia,
        "hora": hora,
        "fin_semana": fin_semana,
        "hora_pico": hora_pico,
        "tavg": temp,
        "tmin": tmin,
        "tmax": tmax,
        "prcp": prcp,
        "wspd": wind,
        "lluvia_bin": lluvia_bin,
        "lluvia_fuerte": lluvia_fuerte,
        "riesgo_frio": riesgo_frio,
        "pav_humedo_frio": pav_humedo_frio
    }])

    df_pred = df_pred[modelo_features]

    prob = modelo_xgb.predict_proba(df_pred)[0, 1]
    return float(prob)


# =============================
#  HERRAMIENTAS PARA EL MODELO
# =============================

def obtener_coordenadas(direccion: str):
    try:
        if "ciudad de méxico" not in direccion.lower() and "cdmx" not in direccion.lower():
            direccion += ", Ciudad de México"

        bounding_box = "19.5921,-98.9496,19.1344,-99.3649"

        geocode_url = (
            f"https://api.tomtom.com/search/2/geocode/{requests.utils.quote(direccion)}.json"
            f"?key={TOMTOM_API}&limit=1&countrySet=MX&btmRight={bounding_box}"
        )

        response = requests.get(geocode_url)
        response.raise_for_status()
        geo_data = response.json()

        if geo_data and geo_data.get("results"):
            position = geo_data["results"][0]["position"]
            return json.dumps({"lat": position["lat"], "lon": position["lon"]})

        return json.dumps({"error": "No se encontraron coordenadas."})

    except Exception as e:
        return json.dumps({"error": str(e)})


def obtener_riesgo_accidente(lat: float, lon: float):
    try:
        # CLIMA
        weather_url = (
            f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
            f"&appid={OWM_API}&units=metric&lang=es"
        )
        w_response = requests.get(weather_url)
        w_response.raise_for_status()
        w = w_response.json()

        clima = w["weather"][0]["description"]
        temp = w["main"]["temp"]
        tmin = w["main"]["temp_min"]
        tmax = w["main"]["temp_max"]
        prcp = w["rain"]["1h"] if "rain" in w else 0
        wind = w["wind"]["speed"]

        # TRÁFICO
        traffic_url = (
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
            f"?point={lat},{lon}&key={TOMTOM_API}"
        )
        t_response = requests.get(traffic_url)
        t_response.raise_for_status()
        t = t_response.json()
        flow = t["flowSegmentData"]
        congestion = 1 - (flow["currentSpeed"] / flow["freeFlowSpeed"]) if flow["freeFlowSpeed"] > 0 else 1.0

        hour = datetime.datetime.now().hour

        # OBTENER ALCALDÍA Y CODIFICAR
        alcaldia = obtener_alcaldia_desde_api(lat, lon)
        try:
            alcaldia_cod = le.transform([alcaldia])[0]
        except:
            alcaldia_cod = 0  # fallback

        # PREDICCIÓN REAL
        riesgo_xgb = predecir_riesgo_modelo(
            alcaldia=alcaldia_cod,
            clima=clima,
            temp=temp,
            tmin=tmin,
            tmax=tmax,
            prcp=prcp,
            wind=wind,
            hora=hour
        )

        return json.dumps({
            "lat": lat,
            "lon": lon,
            "clima": clima,
            "temperatura": temp,
            "tmin": tmin,
            "tmax": tmax,
            "prcp": prcp,
            "viento": wind,
            "congestion": congestion,
            "riesgo_xgb": riesgo_xgb,
            "alcaldia": alcaldia
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================
#  STREAMLIT – UI
# =============================

st.title("🚗 AcciMex — Riesgo de Accidentes en CDMX")
st.caption("Modelo con IA, clima, tráfico y XGBoost integrado.")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "¿Sobre qué zona de la CDMX quieres consultar el riesgo?"
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# =============================
#     CHAT INPUT
# =============================
if prompt := st.chat_input("Escribe una dirección o cruce..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    tools = [
        {"type": "function", "function": {
            "name": "obtener_coordenadas",
            "description": "Convierte dirección a lat/lon",
            "parameters": {"type": "object", "properties": {
                "direccion": {"type": "string"}
            }, "required": ["direccion"]}
        }},
        {"type": "function", "function": {
            "name": "obtener_riesgo_accidente",
            "description": "Calcula tráfico, clima y riesgo",
            "parameters": {"type": "object", "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"}
            }, "required": ["lat", "lon"]}
        }}
    ]

    messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    response = client_openai.chat.completions.create(
        model=model_openai, messages=messages_for_api,
        tools=tools, tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages_for_api.append(response_message)

        for tool_call in tool_calls:
            fn = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Ejecutar herramienta real
            if fn == "obtener_coordenadas":
                result = obtener_coordenadas(**args)
            else:
                result = obtener_riesgo_accidente(**args)

            # Agregar resultado al historial
            messages_for_api.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": fn,
                "content": result
            })

            # Si ya tenemos resultados finales
            result_data = json.loads(result)

            if fn == "obtener_riesgo_accidente":
                with st.chat_message("assistant"):
                    if "error" in result_data:
                        st.error(result_data["error"])
                    else:
                        # ==== MÉTRICAS ====
                        riesgo = result_data["riesgo_xgb"]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("🚨 Riesgo", f"{riesgo*100:.0f}%")
                        col2.metric("🌧️ Lluvia", f"{result_data['prcp']} mm")
                        col3.metric("🚦 Congestión", f"{result_data['congestion']*100:.0f}%")

                        # ==== HEATMAP ====
                        lat = result_data["lat"]
                        lon = result_data["lon"]

                        num_points = 25
                        lats = lat + np.random.normal(0, 0.005, num_points)
                        lons = lon + np.random.normal(0, 0.005, num_points)
                        risk_vals = np.clip(np.random.normal(riesgo, 0.05, num_points), 0, 1)

                        df_heatmap = pd.DataFrame({
                            "lat": lats,
                            "lon": lons,
                            "riesgo": risk_vals
                        })

                        heat_layer = pdk.Layer(
                            "HeatmapLayer",
                            data=df_heatmap,
                            get_position=["lon", "lat"],
                            get_weight="riesgo",
                            radius_pixels=60,
                        )

                        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=13)

                        st.pydeck_chart(pdk.Deck(
                            layers=[heat_layer],
                            initial_view_state=view_state,
                            tooltip={"text": "Riesgo: {riesgo}"}
                        ))

    # SEGUNDA RESPUESTA DEL MODELO
    second_response = client_openai.chat.completions.create(
        model=model_openai,
        messages=messages_for_api
    )

    final_response = second_response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.chat_message("assistant").write(final_response)

else:
    # Si no hubo herramientas ni prompt
    pass
