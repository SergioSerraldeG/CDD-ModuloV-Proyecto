from openai import OpenAI
import requests
import datetime
import json

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
model_openai = "gpt-4o-mini"

OWM_API = os.getenv("OWM_API_KEY")
TOMTOM_API = os.getenv("TOMTOM_API_KEY")

st.title("🚗 AcciMex 🚑")
st.caption(" 🚨 Predicción de riesgo de accidentes automovilisticos en México. 🚑 ")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¿Cómo te puedo ayudar hoy?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Escribe la ubicación o solicitud que tienes aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    conversation = [{"role": "assistant", "content": "Eres un expero en accidentes automovilisticos en la CDMX, en transito y tienes conocimiento de las condiciones climaticas y como impactan en el transito y accidentes automovilisticos. Responde siempre en español."}]
    conversation.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

    with st.chat_message("assistant"):
        stream = client_openai.chat.completions.create(model=model_openai, messages=conversation, stream=True)
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})





@app.get("/riesgo")
def riesgo_accidente(lat: float = 19.4326, lon: float = -99.1332):
    # 1️⃣ Clima
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API}&units=metric"
    w = requests.get(weather_url).json()
    weather = w["weather"][0]["main"]
    temp = w["main"]["temp"]
    wind = w["wind"]["speed"]

    # 2️⃣ Tráfico
    traffic_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={lat},{lon}&key={TOMTOM_API}"
    t = requests.get(traffic_url).json()
    flow = t["flowSegmentData"]
    congestion = 1 - (flow["currentSpeed"] / flow["freeFlowSpeed"])

    # 3️⃣ Riesgo base
    hour = datetime.datetime.now().hour
    riesgo = (
        0.4 * congestion +
        0.3 * (1 if weather in ["Rain", "Thunderstorm"] else 0) +
        0.2 * (wind / 20) +
        0.1 * (1 if hour in [6,7,8,17,18,19] else 0)
    )

    return {
        "lat": lat,
        "lon": lon,
        "clima": weather,
        "temp": temp,
        "viento": wind,
        "congestion": round(congestion, 2),
        "riesgo_accidente": round(min(riesgo, 1), 2)
    }
