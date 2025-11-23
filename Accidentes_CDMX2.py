import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import requests
import datetime
import json
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# CARGAR VARIABLES DE ENTORNO
# ---------------------------
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OWM_API = os.getenv("OWM_API_KEY")
TOMTOM_API = os.getenv("TOMTOM_API_KEY")
TOMTOM_SEARCH_API = os.getenv("TOMTOM_SEARCH_API_KEY")

# ---------------------------
# CARGAR MODELO
# ---------------------------
try:
    model = joblib.load("/Users/mariafernandaserraldegarces/Downloads/Accidentes - Modulo V/modelo_xgb.pkl")
except:
    st.error("❌ No se encontró xgb_model.pkl en el directorio.")
    st.stop()

# ---------------------------
# CONFIGURAR OPENAI
# ---------------------------
client_openai = OpenAI(api_key=OPENAI_API_KEY)
model_openai = "gpt-4o-mini"

# ---------------------------
#   1. OBTENER COORDENADAS
# ---------------------------
def obtener_coordenadas(direccion: str):
    try:
        if "cdmx" not in direccion.lower():
            direccion += ", Ciudad de México"

        direccion_encoded = requests.utils.quote(direccion)

        url = (
            f"https://api.tomtom.com/search/2/geocode/{direccion_encoded}.json"
            f"?key={TOMTOM_SEARCH_API}&limit=1&countrySet=MX"
        )

        resp = requests.get(url).json()
        if resp.get("results"):
            pos = resp["results"][0]["position"]
            return json.dumps({"lat": pos["lat"], "lon": pos["lon"]})

        return json.dumps({"error": "No se encontraron coordenadas."})

    except Exception as e:
        return json.dumps({"error": f"Error obteniendo coordenadas: {e}"})


# ---------------------------
#   2. OBTENER FEATURES Y RIESGO
# ---------------------------
def obtener_riesgo_accidente(lat: float, lon: float):
    try:
        # 🌦️ API CLIMA
        url_weather = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OWM_API}&units=metric"
        )
        w = requests.get(url_weather).json()

        clima = {
            "temperature": w["main"]["temp"],
            "humidity": w["main"]["humidity"],
            "pressure": w["main"]["pressure"],
            "wind_speed": w["wind"]["speed"],
            "cloudiness": w["clouds"]["all"],
            "visibility": w.get("visibility", 10000)
        }

        # 🚗 API TRÁFICO
        url_traffic = (
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            f"?point={lat},{lon}&key={TOMTOM_API}"
        )
        t = requests.get(url_traffic).json().get("flowSegmentData", {})

        trafico = {
            "current_speed": t.get("currentSpeed", 0),
            "free_flow_speed": t.get("freeFlowSpeed", 1),
            "current_travel_time": t.get("currentTravelTime", 0),
            "free_flow_travel_time": t.get("freeFlowTravelTime", 0),
            "confidence": t.get("confidence", 0),
        }

        # Tiempo
        now = datetime.datetime.now()
        clima.update({
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0
        })

        # Construir vector
        vector = np.array([
            clima["temperature"], clima["humidity"], clima["pressure"],
            clima["wind_speed"], clima["cloudiness"], clima["visibility"],
            clima["hour"], clima["day_of_week"], clima["is_weekend"],
            trafico["current_speed"], trafico["free_flow_speed"],
            trafico["current_travel_time"], trafico["free_flow_travel_time"],
            trafico["confidence"]
        ]).reshape(1, -1)

        # 🔮 Predicción
        riesgo = float(model.predict(vector)[0])

        return json.dumps({
            "lat": lat,
            "lon": lon,
            "temperatura": clima["temperature"],
            "congestion": 1 - (trafico["current_speed"] / trafico["free_flow_speed"]),
            "riesgo_calculado": riesgo
        })

    except Exception as e:
        return json.dumps({"error": f"Error al calcular riesgo: {e}"})


# ---------------------------
#      INTERFAZ STREAMLIT
# ---------------------------

st.title("🚗 AcciMex — Riesgo Vial Inteligente")
st.caption("Asistente inteligente con IA, clima, tráfico y modelo predictivo real.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¿Sobre qué zona de la CDMX quieres consultar el riesgo? Puedes escribir una calle, cruce o colonia."}
    ]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

prompt = st.chat_input("Ej. 'Reforma e Insurgentes'")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "obtener_coordenadas",
                "description": "Devuelve lat y lon para una dirección.",
                "parameters": {
                    "type": "object",
                    "properties": {"direccion": {"type": "string"}},
                    "required": ["direccion"]
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "obtener_riesgo_accidente",
                "description": "Calcula riesgo real usando APIs + modelo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"}
                    },
                    "required": ["lat", "lon"]
                },
            },
        }
    ]

    messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # 1️⃣ Llamada principal
    response = client_openai.chat.completions.create(
        model=model_openai, messages=messages_for_api, tools=tools, tool_choice="auto"
    )

    msg = response.choices[0].message
    tool_calls = msg.tool_calls

    if tool_calls:
        messages_for_api.append(msg)

        for call in tool_calls:
            fname = call.function.name
            args = json.loads(call.function.arguments)

            if fname == "obtener_coordenadas":
                res = obtener_coordenadas(**args)

            elif fname == "obtener_riesgo_accidente":
                res = obtener_riesgo_accidente(**args)

                # Mostrar dashboard visual (igual que tu versión)
                data = json.loads(res)
                with st.chat_message("assistant"):
                    st.write("Aquí tienes un análisis visual de la zona:")
                    col1, col2, col3 = st.columns(3)

                    col1.metric("🚨 Riesgo", f"{data['riesgo_calculado']*100:.1f}%")
                    col2.metric("🚦 Congestión", f"{data['congestion']*100:.0f}%")
                    col3.metric("🌡️ Temperatura", f"{data['temperatura']:.1f}°C")

                    st.map(pd.DataFrame([{"lat": data["lat"], "lon": data["lon"]}]), zoom=13)

            messages_for_api.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fname,
                "content": res
            })

        # Segunda llamada para generar texto final
        final = client_openai.chat.completions.create(
            model=model_openai,
            messages=messages_for_api,
        )

        text = final.choices[0].message.content

    else:
        text = msg.content

    st.session_state.messages.append({"role": "assistant", "content": text})
    st.chat_message("assistant").write(text)
