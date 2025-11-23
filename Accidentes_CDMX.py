import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import requests
import datetime
import json
import pandas as pd


# Cargar variables de entorno desde el archivo .env
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OWM_API = os.getenv("OWM_API_KEY")
TOMTOM_API = os.getenv("TOMTOM_API_KEY")

# Configurar cliente de OpenAI
client_openai = OpenAI(api_key=OPENAI_API_KEY)
model_openai = "gpt-4o-mini"

# --- Herramienta 1: Función de Geocodificación ---
def obtener_coordenadas(direccion: str):
    """
    Convierte una dirección o nombre de lugar en coordenadas (latitud, longitud) usando una API de geocodificación.
    Añade ", Ciudad de México" para mejorar la precisión de la búsqueda.
    """
    try:
        if "ciudad de méxico" not in direccion.lower() and "cdmx" not in direccion.lower():
            direccion += ", Ciudad de México"
            
         
        # --- MEJORA DE PRECISIÓN ---
        # Añadimos un "bounding box" para la CDMX y restringimos la búsqueda a México.
        # Esto le da una fuerte prioridad a los resultados dentro de la ciudad.
        # Coordenadas del cuadro: Top-Left (Noreste), Bottom-Right (Suroeste)
        bounding_box = "19.5921,-98.9496,19.1344,-99.3649" # Formato: maxLat,maxLon,minLat,minLon
        
        geocode_url = f"https://api.tomtom.com/search/2/geocode/{requests.utils.quote(direccion)}.json?key={TOMTOM_API}&limit=1&countrySet=MX&btmRight={bounding_box}"
        response = requests.get(geocode_url)
        response.raise_for_status()
        geo_data = response.json()
        
        if geo_data and geo_data.get("results"):
            position = geo_data["results"][0]["position"]
            return json.dumps({"lat": position["lat"], "lon": position["lon"]})
        return json.dumps({"error": "No se encontraron coordenadas para la ubicación especificada."})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Error en la API de geocodificación: {e}"})

# --- Herramienta 2: Función de Cálculo de Riesgo ---
def obtener_riesgo_accidente(lat: float = 19.4326, lon: float = -99.1332):
    """
    Obtiene el clima actual, el tráfico y calcula un riesgo de accidente para una ubicación específica.
    Usa CDMX como ubicación por defecto si no se proporcionan coordenadas.
    """
    try:
        # 1. Clima (OpenWeatherMap)
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API}&units=metric&lang=es"
        w_response = requests.get(weather_url)
        w_response.raise_for_status()
        w = w_response.json()
        weather = w["weather"][0]["description"]
        temp = w["main"]["temp"]
        wind = w["wind"]["speed"]

        # 2. Tráfico (TomTom)
        traffic_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={lat},{lon}&key={TOMTOM_API}"
        t_response = requests.get(traffic_url)
        t_response.raise_for_status()
        t = t_response.json()
        flow = t["flowSegmentData"]
        congestion = 1 - (flow["currentSpeed"] / flow["freeFlowSpeed"]) if flow["freeFlowSpeed"] > 0 else 1.0

        # 3. Riesgo base (lógica simplificada)
        hour = datetime.datetime.now().hour
        riesgo = (
            0.4 * congestion +
            0.3 * (1 if "rain" in weather or "thunderstorm" in weather else 0) +
            0.2 * (wind / 20) +
            0.1 * (1 if hour in [6, 7, 8, 17, 18, 19] else 0)
        )

        return json.dumps({
            "lat": lat, "lon": lon, "clima": weather, "temperatura": temp,
            "viento_kmh": wind * 3.6, "congestion": round(congestion, 2),
            "riesgo_calculado": round(min(riesgo, 1), 2)
        })
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"No se pudieron obtener los datos externos: {e}"})
    except KeyError as e:
        return json.dumps({"error": f"La respuesta de la API no tuvo el formato esperado. Faltó la clave: {e}"})

# --- Interfaz de Streamlit ---

# Inyectar CSS para cambiar el color de fondo
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 AcciMex 🚑")
st.caption("🚨 Asistente de riesgo de accidentes viales en CDMX.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿Sobre qué zona de la CDMX quieres consultar el riesgo? Para mayor precisión, intenta con una dirección, un cruce de calles o una colonia."}]

# Mostrar mensajes del historial
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Lógica principal del chat
if prompt := st.chat_input(placeholder="Escribe una dirección, cruce o colonia (ej. 'Reforma e Insurgentes')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Definir las herramientas disponibles para el modelo
    tools = [
        {"type": "function", "function": {
            "name": "obtener_coordenadas",
            "description": "Obtiene la latitud y longitud para una dirección o nombre de lugar proporcionado por el usuario.",
            "parameters": {"type": "object", "properties": {
                "direccion": {"type": "string", "description": "El nombre del lugar o dirección, ej: 'Colonia Roma' o 'Paseo de la Reforma 222'."},
            }, "required": ["direccion"]},
        }},
        {"type": "function", "function": {
            "name": "obtener_riesgo_accidente",
            "description": "Obtiene clima, tráfico y riesgo de accidente para una latitud y longitud específicas en CDMX.",
            "parameters": {"type": "object", "properties": {
                "lat": {"type": "number", "description": "La latitud de la ubicación a consultar."},
                "lon": {"type": "number", "description": "La longitud de la ubicación a consultar."}
            }, "required": ["lat", "lon"]},
        }},
    ]
    
    messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    
    # Primera llamada a la API
    response = client_openai.chat.completions.create(
        model=model_openai, messages=messages_for_api, tools=tools, tool_choice="auto"
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Si el modelo decide usar una herramienta
    if tool_calls:
        messages_for_api.append(response_message)
        
        # Bucle para manejar múltiples llamadas a herramientas en secuencia
        while tool_calls:
            available_functions = {"obtener_riesgo_accidente": obtener_riesgo_accidente, "obtener_coordenadas": obtener_coordenadas}
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                
                # Si se llamó a la función de riesgo, mostrar el dashboard visual
                if function_name == "obtener_riesgo_accidente":
                    with st.chat_message("assistant"):
                        st.write("Aquí tienes un análisis visual de la zona:")
                        response_data = json.loads(function_response)
                        if "error" not in response_data:
                            col1, col2, col3 = st.columns(3)
                            riesgo_val = response_data.get('riesgo_calculado', 0)
                            col1.metric(label="🚨 Nivel de Riesgo", value=f"{riesgo_val * 100:.0f}%", 
                                        delta=f"{'Alto' if riesgo_val > 0.6 else 'Moderado' if riesgo_val > 0.3 else 'Bajo'}",
                                        delta_color="inverse")
                            col2.metric(label="🚦 Congestión", value=f"{response_data.get('congestion', 0) * 100:.0f}%")
                            col3.metric(label="🌡️ Temperatura", value=f"{response_data.get('temperatura', 0):.1f} °C")
                            map_data = pd.DataFrame({'lat': [response_data.get('lat')], 'lon': [response_data.get('lon')]})
                            st.map(map_data, zoom=13)

                messages_for_api.append(
                    {"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response}
                )
            
            # Siguiente llamada para que el modelo decida qué hacer con la nueva información
            response = client_openai.chat.completions.create(
                model=model_openai, messages=messages_for_api, tools=tools, tool_choice="auto"
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # antes de que el bucle continúe o termine.
            if tool_calls:
                messages_for_api.append(response_message)

        final_response = response_message.content
    else:
        # Si el modelo no necesitó herramientas, usar su respuesta directamente
        final_response = response_message.content

    # Mostrar la respuesta final de texto del asistente
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.chat_message("assistant").info(final_response)
