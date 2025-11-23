import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from unidecode import unidecode
import requests
import datetime
import json
import pandas as pd
import joblib
import numpy as np
import pydeck as pdk
from sklearn.preprocessing import LabelEncoder
import time

# --- CARGA DE VARIABLES Y MODELO ---

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OWM_API = os.getenv("OWM_API_KEY")
TOMTOM_API = os.getenv("TOMTOM_API_KEY")

client_openai = None
if OPENAI_API_KEY:
    try:
        client_openai = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"No se pudo inicializar el cliente de OpenAI: {e}")

model_openai = "gpt-4o-mini"

try:
    # Carga el modelo, el codificador y la lista de features desde el archivo .pkl
    modelo_data = joblib.load("/Users/mariafernandaserraldegarces/Downloads/Accidentes - Modulo V/modelo_xgb_full.pkl")
    modelo_xgb = modelo_data.get("model")
    modelo_features = modelo_data.get("features")
    le_alcaldia = LabelEncoder()
    alcaldias_conocidas = [
        'GUSTAVO A MADERO','IZTAPALAPA','CUAUHTEMOC','BENITO JUAREZ','MIGUEL HIDALGO',
        'COYOACAN','ALVARO OBREGON','VENUSTIANO CARRANZA','AZCAPOTZALCO',
        'IZTACALCO','TLALPAN','TLAHUAC','XOCHIMILCO','CUAJIMALPA DE MORELOS',
        'LA MAGDALENA CONTRERAS','MILPA ALTA'
    ]
    le_alcaldia.fit(alcaldias_conocidas)

    if modelo_xgb is None or modelo_features is None:
        st.error("El archivo pickle debe contener 'model' y 'features'.")
        st.stop()
except FileNotFoundError:
    st.error("Error: No se encontró 'modelo_riesgo_xgb.pkl'. Asegúrate de que esté en el mismo directorio.")
    st.stop()
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

# --- DICCIONARIO DE COORDENADAS PARA MAPA Y HERRAMIENTAS ---
coordenadas_alcaldias = {
    'AZCAPOTZALCO': (19.4842, -99.1859), 'COYOACAN': (19.3297, -99.1603),
    'CUAJIMALPA DE MORELOS': (19.3636, -99.2933), 'GUSTAVO A MADERO': (19.5048, -99.1115),
    'IZTACALCO': (19.3952, -99.0943), 'IZTAPALAPA': (19.3553, -99.0622),
    'LA MAGDALENA CONTRERAS': (19.3037, -99.2435), 'MILPA ALTA': (19.1922, -99.0231),
    'ALVARO OBREGON': (19.359, -99.207), 'TLAHUAC': (19.263, -99.004),
    'TLALPAN': (19.288, -99.165), 'XOCHIMILCO': (19.255, -99.105),
    'BENITO JUAREZ': (19.381, -99.165), 'CUAUHTEMOC': (19.435, -99.145),
    'MIGUEL HIDALGO': (19.432, -99.192), 'VENUSTIANO CARRANZA': (19.42, -99.105)
}

# --- FUNCIONES DE HERRAMIENTAS ---

def obtener_coordenadas(direccion: str):
    """Convierte una dirección o nombre de lugar en coordenadas (latitud, longitud)."""
    try:
        if "ciudad de méxico" not in direccion.lower() and "cdmx" not in direccion.lower():
            direccion += ", Ciudad de México"
        
        bounding_box = "19.5921,-98.9496,19.1344,-99.3649"
        geocode_url = f"https://api.tomtom.com/search/2/geocode/{requests.utils.quote(direccion)}.json?key={TOMTOM_API}&limit=1&countrySet=MX&btmRight={bounding_box}"
        
        resp = requests.get(geocode_url, timeout=10)
        resp.raise_for_status()
        geo_data = resp.json()

        if geo_data and geo_data.get("results"):
            pos = geo_data["results"][0]["position"]
            return json.dumps({"lat": pos["lat"], "lon": pos["lon"]})
        return json.dumps({"error": "No se encontraron coordenadas para la ubicación especificada."})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Error en la API de geocodificación: {e}"})

def obtener_riesgo_accidente(lat: float, lon: float):
    """Obtiene datos de clima/tráfico y predice el riesgo de accidente con el modelo XGBoost."""
    try:
        # 1. Clima
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API}&units=metric&lang=es"
        w_resp = requests.get(weather_url, timeout=10)
        w_resp.raise_for_status()
        w = w_resp.json()

        weather = w["weather"][0]["description"]
        clima = w["weather"][0]["description"]
        temp = w["main"]["temp"]
        tmin = w["main"]["temp_min"]
        tmax = w["main"]["temp_max"]
        prcp = w.get("rain", {}).get("1h", 0)
        wind = w.get("wind", {}).get("speed", 0)

        # 2. Tráfico
        traffic_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={lat},{lon}&key={TOMTOM_API}"
        t_resp = requests.get(traffic_url, timeout=10)
        t_resp.raise_for_status()
        t = t_resp.json()
        flow = t.get("flowSegmentData", {})
        congestion = 1 - (flow.get("currentSpeed", 0) / flow.get("freeFlowSpeed", 1)) if flow.get("freeFlowSpeed", 0) > 0 else 1.0
        congestion = 1 - (flow.get("currentSpeed", 0) / flow.get("freeFlowSpeed", 1)) if flow.get("freeFlowSpeed", 1) > 0 else 1.0

        # 3. Alcaldía
        geocode_url = f"https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json?key={TOMTOM_API}"
        geo_resp = requests.get(geocode_url, timeout=10).json()
        address_details = geo_resp.get('addresses', [{}])[0].get('address', {})
        alcaldia_actual = address_details.get('municipality', 'CUAUHTEMOC')
        alcaldia_norm = unidecode(alcaldia_actual).strip().upper()
        # 3. Riesgo (Lógica simplificada como en Accidentes_CDMX.py)
        hour = datetime.datetime.now().hour
        riesgo = (
            0.4 * congestion +
            0.3 * (1 if "rain" in clima or "thunderstorm" in clima else 0) +
            0.2 * (wind / 20) +
            0.1 * (1 if hour in [6, 7, 8, 17, 18, 19] else 0)
        )
        riesgo_final = round(min(riesgo, 1.0), 2)

        if alcaldia_norm not in le_alcaldia.classes_:
            alcaldia_norm = 'CUAUHTEMOC'

        # 4. Ingeniería de Features
        now = datetime.datetime.now()
        hour = now.hour
        data_para_predecir = {
            # --- CORRECCIÓN: Codificar la alcaldía a su valor numérico ---
            # El modelo XGBoost fue entrenado con números para las alcaldías,
            # no con texto. Esta línea convierte el nombre (ej. "COYOACAN")
            # al número correspondiente que el modelo entiende.
            'alcaldia': int(le_alcaldia.transform([alcaldia_norm])[0]),
            'hora': hour,
            'fin_semana': 1 if now.weekday() >= 5 else 0,
            'hora_pico': 1 if (6 <= hour <= 9) or (17 <= hour <= 20) else 0,
            'tavg': temp, 'tmin': tmin, 'tmax': tmax, 'prcp': prcp, 'wspd': wind,
            'lluvia_bin': 1 if prcp > 0 else 0,
            'lluvia_fuerte': 1 if prcp >= 10 else 0,
            'riesgo_frio': 1 if tmin <= 10 else 0,
            'pav_humedo_frio': 1 if (tmin <= 10 and prcp > 0) else 0
        }
        df_pred = pd.DataFrame([data_para_predecir])[modelo_features]

        # 5. Predicción
        riesgo_base_modelo = float(modelo_xgb.predict_proba(df_pred)[0, 1])

        # --- MEJORA: Ajuste por Congestión Específica ---
        # Usamos la congestión del punto exacto como un factor de ajuste.
        # Si la congestión es alta (cercana a 1), el riesgo aumenta.
        # Si es baja (cercana a 0), el riesgo disminuye.
        # El factor 0.2 modera el impacto para no desviar demasiado la predicción base.
        riesgo_ajustado = riesgo_base_modelo * (1 + (congestion * 0.5)) # Aumentamos el impacto del ajuste
        riesgo_final = min(riesgo_ajustado, 0.99) # Aseguramos que no pase de 99%

        return json.dumps({
            "lat": lat, "lon": lon, "clima": weather, "temperatura": temp,
            "lat": lat, "lon": lon, "clima": clima, "temperatura": temp,
            "congestion": round(congestion, 2), "alcaldia": alcaldia_actual,
            "riesgo_calculado": riesgo_final
        })
    except Exception as e:
        return json.dumps({"error": f"Error al calcular riesgo: {e}"})

def obtener_riesgo_especifico(lat: float, lon: float):
    """
    Calcula el riesgo para una ubicación específica usando una fórmula ponderada,
    similar a la del archivo Accidentes_CDMX.py.
    """
    try:
        # 1. Clima (OpenWeatherMap)
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API}&units=metric&lang=es"
        w_resp = requests.get(weather_url, timeout=10)
        w_resp.raise_for_status()
        w = w_resp.json()
        clima = w["weather"][0]["description"]
        temp = w["main"]["temp"]
        wind = w["wind"]["speed"]

        # 2. Tráfico (TomTom)
        traffic_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={lat},{lon}&key={TOMTOM_API}"
        t_resp = requests.get(traffic_url, timeout=10)
        t_resp.raise_for_status()
        t = t_resp.json()
        flow = t.get("flowSegmentData", {})
        congestion = 1 - (flow.get("currentSpeed", 0) / flow.get("freeFlowSpeed", 1)) if flow.get("freeFlowSpeed", 1) > 0 else 1.0

        # 3. Riesgo (Lógica simplificada y ponderada)
        hour = datetime.datetime.now().hour
        riesgo = (
            0.4 * congestion +
            0.3 * (1 if "rain" in clima or "thunderstorm" in clima else 0) +
            0.2 * (wind / 20) +
            0.1 * (1 if hour in [6, 7, 8, 17, 18, 19] else 0)
        )
        riesgo_final = round(min(riesgo, 1.0), 2)

        return json.dumps({"lat": lat, "lon": lon, "clima": clima, "temperatura": temp, "congestion": round(congestion, 2), "riesgo_calculado": riesgo_final})
    except Exception as e:
        return json.dumps({"error": f"Error al calcular riesgo específico: {e}"})

def encontrar_alcaldia_mayor_riesgo():
    """
    Calcula el riesgo para todas las alcaldías y devuelve la que tiene el mayor riesgo.
    Útil para preguntas como '¿cuál es la zona de mayor riesgo?'. No necesita argumentos.
    """
    riesgos = []
    for alcaldia, (lat, lon) in coordenadas_alcaldias.items():
        try:
            resp_str = obtener_riesgo_accidente(lat, lon)
            resp_data = json.loads(resp_str)
            riesgo = resp_data.get("riesgo_calculado", 0.0) if "error" not in resp_data else 0.0
            riesgos.append({"alcaldia": alcaldia, "riesgo": riesgo})
        except Exception:
            riesgos.append({"alcaldia": alcaldia, "riesgo": 0.0})
        time.sleep(0.1)

    if not riesgos:
        return json.dumps({"error": "No se pudo calcular el riesgo para ninguna alcaldía."})

    alcaldia_max_riesgo = max(riesgos, key=lambda x: x['riesgo'])
    nombre = alcaldia_max_riesgo['alcaldia']
    valor = alcaldia_max_riesgo['riesgo']
    
    return json.dumps({
        "mensaje": f"Actualmente, la alcaldía con mayor riesgo es {nombre} con una probabilidad de accidente del {valor:.1%}."
    })

# --- MAPA DE CALOR POR ALCALDÍA ---
@st.cache_data(ttl=900)
def generar_mapa_calor_alcaldias(return_df=False):
    """
    Calcula el riesgo para cada alcaldía y genera un mapa coroplético con PyDeck.
    Si return_df es True, devuelve el DataFrame de riesgos en lugar del mapa.
    """
    riesgos = []
    for alcaldia, (lat, lon) in coordenadas_alcaldias.items():
        try:
            resp_str = obtener_riesgo_accidente(lat, lon)
            resp_data = json.loads(resp_str)
            riesgo = resp_data.get("riesgo_calculado", 0.0) if "error" not in resp_data else 0.0
            riesgos.append({"alcaldia": alcaldia, "riesgo_calculado": riesgo})
        except Exception as e:
            print(f"Error calculando riesgo para {alcaldia}: {e}")
            riesgos.append({"alcaldia": alcaldia, "riesgo_calculado": 0.0})
        time.sleep(0.1) # Pequeña pausa para no saturar APIs
    
    df_riesgos = pd.DataFrame(riesgos)

    if return_df:
        return df_riesgos

    try:
        with open("alcaldias_cdmx.geojson", "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error("No se encontró 'alcaldias_cdmx.geojson'. Asegúrate de que el archivo esté en el directorio del proyecto.")
        return None

    # --- MEJORA: Escala de color dinámica ---
    # 1. Encontrar el riesgo mínimo y máximo para normalizar la escala de colores.
    min_riesgo = df_riesgos['riesgo_calculado'].min()
    max_riesgo = df_riesgos['riesgo_calculado'].max()
    rango_riesgo = max_riesgo - min_riesgo if max_riesgo > min_riesgo else 1.0

    for feature in geojson_data["features"]:
        nombre_original = feature["properties"].get("NOMGEO", "")
        nombre_normalizado = unidecode(nombre_original).strip().upper()
        
        row = df_riesgos[df_riesgos["alcaldia"] == nombre_normalizado]
        
        riesgo_actual = float(row.iloc[0]["riesgo_calculado"]) if not row.empty else 0.0
        feature["properties"]["riesgo_calculado"] = riesgo_actual

        # 2. Normalizar el riesgo actual a una escala de 0 a 1 y calcular el color.
        riesgo_normalizado = (riesgo_actual - min_riesgo) / rango_riesgo
        r = int(255 * riesgo_normalizado)
        g = int(255 * (1 - riesgo_normalizado))
        b = 0
        feature["properties"]["color"] = [r, g, b]

    view_state = pdk.ViewState(latitude=19.4, longitude=-99.13, zoom=9.5, bearing=0, pitch=45)
    layer = pdk.Layer(
        "GeoJsonLayer", geojson_data, opacity=0.5, stroked=True, filled=True,
        get_fill_color="properties.color", get_line_color=[255, 255, 255],
        get_line_width=100, pickable=True
    )
    tooltip = {"html": "<b>Alcaldía:</b> {NOMGEO}<br><b>Riesgo:</b> {riesgo_calculado}"}
    
    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v9", tooltip=tooltip)

# -------------------------------------------------------------
# --------------------- INTERFAZ STREAMLIT ---------------------
# -------------------------------------------------------------

# --- Estilo general del fondo ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .titulo_principal {
            text-align: center;
            font-size: 42px;
            color: #1F2937;
            font-weight: 700;
            margin-bottom: -10px;
        }
        .subtitulo_principal {
            text-align: center;
            font-size: 18px;
            color: #374151;
        }
        .linea-titulo {
            border: none;
            border-top: 2px solid #3B82F6;
            width: 60%;
            margin: 0 auto 25px auto;
        }
        .search-input {
            width: 100%;
            padding: 15px;
            font-size: 18px;
            border-radius: 12px;
            border: 1px solid #ccc;
            margin-top: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------ HEADER CON LOGO MODERNO -------------------
import base64

def render_header():
    try:
        logo_path = "/Users/mariafernandaserraldegarces/Desktop/SGSG/CDD-ModuloV-Proyecto/SafeRoad CDMX - Modulo V.png"

        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()

        st.markdown("""
            <style>
                .titulo_principal {
                    color: #0d3a70 !important;    /* Azul marino */
                    text-align: center !important;
                    font-weight: 900 !important;
                }
                .subtitulo_principal {
                    color: #FF8C42 !important;   /* Naranja */
                    text-align: center !important;
                    font-size: 18px !important;
                    margin-top: -10px !important;
                }
                .linea-titulo {
                    border: 1px solid #ccc;
                    margin-top: 15px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
    <div style="text-align:center; margin-top:-40px; margin-bottom:5px;">
        <img src="data:image/png;base64,{logo_base64}" style="width:170px;">
    </div>

    <h1 class="titulo_principal">🚗 SafeRoad CDMX 🚑</h1>
    <p class="subtitulo_principal">Asistente de riesgo de accidentes viales en CDMX con IA.</p>
    <hr class="linea-titulo">
""", unsafe_allow_html=True)


    except:
        st.warning("No se pudo cargar el logo.")


render_header()


# ------------------- TÍTULO DE SECCIÓN ------------------------
def titulo_section(texto):
    st.markdown(f"""
    <h3 style="color:#1F2937; margin-bottom:0px; margin-top:30px;">{texto}</h3>
    <hr style="border:1px solid #93C5FD; margin-top:0px; margin-bottom:15px;">
    """, unsafe_allow_html=True)


# ------------------- MAPA PRINCIPAL ---------------------------
titulo_section("Mapa de Riesgo por Alcaldía en Tiempo Real")

mapa_deck = generar_mapa_calor_alcaldias()
if mapa_deck:
    st.pydeck_chart(mapa_deck)


# ------------------- CHAT + BUSCADOR ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "¿Sobre qué zona de la CDMX quieres consultar el riesgo? "
            "Puedes ingresar una dirección, un cruce o una colonia."
        )
    }]

# Mostrar historial
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ------- BARRA DE CHAT STICKY --------
# Reemplazamos st.text_input por st.chat_input.
# Este widget está diseñado para permanecer en la parte inferior de la pantalla.
if prompt := st.chat_input("🔍 Escribe una dirección, cruce o colonia…"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ----- RESTO DE TU LÓGICA -----
    tools = [
        {"type": "function", "function": {
            "name": "obtener_coordenadas",
            "description": "Obtiene la latitud y longitud para una dirección, calle o colonia específica.",
            "parameters": {"type": "object", "properties": {
                "direccion": {"type": "string"},
            }, "required": ["direccion"]},
        }},
        {"type": "function", "function": {
            "name": "encontrar_alcaldia_mayor_riesgo",
            "description": "Encuentra la alcaldía con el mayor riesgo actual.",
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "obtener_riesgo_especifico",
            "description": "Calcula el riesgo usando clima + tráfico para un punto.",
            "parameters": {"type": "object", "properties": {
                "lat": {"type": "number"}, "lon": {"type": "number"}
            }, "required": ["lat", "lon"]},
        }}
    ]

    messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.spinner("Analizando tu solicitud..."):
        response = client_openai.chat.completions.create(
            model=model_openai, messages=messages_for_api, tools=tools, tool_choice="auto"
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "obtener_coordenadas":
                with st.spinner("Obteniendo coordenadas y calculando riesgo..."):
                    coord_resp_str = obtener_coordenadas(**function_args)
                    coord_resp = json.loads(coord_resp_str)
                    
                    if "error" in coord_resp:
                        st.chat_message("assistant").error(f"Error: {coord_resp['error']}")
                    else:
                        lat, lon = coord_resp["lat"], coord_resp["lon"]
                        risk_resp_str = obtener_riesgo_especifico(lat, lon) # <-- LLAMADA A LA NUEVA FUNCIÓN
                        risk_resp = json.loads(risk_resp_str)

                        if "error" in risk_resp:
                            st.chat_message("assistant").error(f"Error: {risk_resp['error']}")
                        else:
                            with st.chat_message("assistant"):
                                riesgo_val = risk_resp.get("riesgo_calculado", 0.0)
                                
                                # --- MEJORA: UMBRALES DINÁMICOS ---
                                df_riesgos_mapa = generar_mapa_calor_alcaldias(return_df=True)
                                umbral_moderado = df_riesgos_mapa['riesgo_calculado'].quantile(0.50)
                                umbral_alto = df_riesgos_mapa['riesgo_calculado'].quantile(0.80)

                                st.write(f"Análisis para: **{risk_resp.get('alcaldia', 'Ubicación')}**")
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric(label="🚨 Nivel de Riesgo", value=f"{riesgo_val * 100:.1f}%", 
                                            delta=f"{'Alto' if riesgo_val >= umbral_alto else 'Moderado' if riesgo_val >= umbral_moderado else 'Bajo'}",
                                            delta_color="inverse")
                                col2.metric(label="🚦 Congestión", value=f"{risk_resp.get('congestion', 0) * 100:.0f}%")
                                col3.metric(label="🌡️ Temperatura", value=f"{risk_resp.get('temperatura', 0):.1f}°C")
                                
                                # --- MAPA SIMPLE CON PIN ---
                                # Reemplazamos el heatmap por un mapa simple que muestra la ubicación exacta.
                                map_data = pd.DataFrame([{'lat': lat, 'lon': lon}])
                                st.map(map_data, zoom=14, color='#FF4B4B')

                                summary_prompt = (
                                    f"Actúa como un asistente de seguridad vial. Resume el siguiente análisis de riesgo en un párrafo corto y amigable. "
                                    f"Datos: Riesgo calculado={riesgo_val:.2%}, Temperatura={risk_resp['temperatura']:.1f}°C, "
                                    f"Condición climática='{risk_resp['clima']}', Congestión={risk_resp['congestion']:.0%}. "
                                    f"Ofrece una recomendación de manejo basada en estos datos."
                                )
                                response = client_openai.chat.completions.create(
                                    model=model_openai, messages=[{"role": "user", "content": summary_prompt}], temperature=0.5
                                )
                                final_text = response.choices[0].message.content
                                st.info(final_text)
                                st.session_state.messages.append({"role": "assistant", "content": final_text})

            elif function_name == "encontrar_alcaldia_mayor_riesgo":
                with st.spinner("Calculando riesgo para todas las alcaldías..."):
                    result_str = encontrar_alcaldia_mayor_riesgo()
                    result_data = json.loads(result_str)
                    final_text = result_data.get("mensaje", "No se pudo determinar la alcaldía de mayor riesgo.")
                    st.chat_message("assistant").info(final_text)
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
        
        else: # Si el modelo responde directamente sin usar herramientas
            final_text = response_message.content
            st.chat_message("assistant").write(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
