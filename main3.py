import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from prompts import stronger_prompt

# Cargar variables de entorno desde el archivo .env
load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configurar el cliente de Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-pro")

st.title("📊 FinguIA")
st.caption("💰 Inversiones simplificadas.")

# Inicializar el historial de mensajes en el estado de la sesión si no existe
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Escribe tu mensaje aquí..."):
    # Añadir el mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Preparar el historial para la API de Gemini
    # El rol 'assistant' se corresponde con el rol 'model' en Gemini.
    gemini_history = [
        {"role": "model" if msg["role"] == "assistant" else "user", "parts": [msg["content"]]}
        for msg in st.session_state.messages[:-1]  # Excluir el último mensaje del usuario
    ]

    # Generar la respuesta del modelo en streaming
    with st.chat_message("assistant"):
        # Iniciar una sesión de chat con el historial
        chat = model_gemini.start_chat(history=gemini_history)
        # Enviar el nuevo mensaje del usuario junto con la instrucción del sistema (stronger_prompt)
        stream = chat.send_message(f"{stronger_prompt}\n\n{prompt}", stream=True)
        
        # st.write_stream espera un generador de strings, no de objetos complejos.
        # Este generador extrae el texto de cada trozo que envía la API de Gemini.
        # Se añade un `try-except` para ignorar los trozos vacíos al final del stream que causan el ValueError.
        def stream_generator(stream):
            for chunk in stream:
                try:
                    yield chunk.text
                except ValueError:
                    pass # Ignora los trozos que no tienen texto.
        response = st.write_stream(stream_generator(stream))

    # Añadir la respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": response})