import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Cargar variables de entorno desde el archivo .env
load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configurar el cliente de Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel("models/gemini-2.5-pro") # <-- Pega aquí el nombre correcto

st.title("📊 FinguIA")
st.caption("💰 Inversiones simplificadas.")

# Inicializar el historial de mensajes en el estado de la sesión si no existe
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]

# Mostrar los mensajes existentes en el historial
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Capturar la entrada del usuario desde el campo de chat
if prompt := st.chat_input(placeholder="Escribe tu mensaje aquí..."):
    # Añadir el mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Preparar el historial para la API de Gemini (cambiando 'assistant' por 'model')
    # El rol 'assistant' de OpenAI se corresponde con el rol 'model' en Gemini.
    gemini_history = [
        {"role": "model" if msg["role"] == "assistant" else "user", "parts": [msg["content"]]}
        for msg in st.session_state.messages[:-1]  # Excluir el último mensaje del usuario
    ]

    # Generar la respuesta del modelo en streaming
    with st.chat_message("assistant"):
        # Iniciar una sesión de chat con el historial
        chat = model_gemini.start_chat(history=gemini_history)
        # Enviar el nuevo mensaje del usuario (el prompt actual) junto con la instrucción del sistema
        system_prompt = "Eres un experto en inversiones y en la bolsa de valores. Responde siempre en español."
        stream = chat.send_message(f"{system_prompt}\n\n{prompt}", stream=True)
        response = st.write_stream(stream)

    # Añadir la respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": response})