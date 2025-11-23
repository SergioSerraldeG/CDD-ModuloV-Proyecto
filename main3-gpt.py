import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from prompts import stronger_prompt

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

model_openai = "gpt-5-mini"
model_deepseek = "deepseek-chat"

st.title("📊 FinguIA")
st.caption("💰 Inversiones simplificadas.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Escribe tu mensaje aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    conversation = [{"role": "assistant", "content": stronger_prompt}]
    conversation.extend({"role": m["role"], "content":m["content"]} for m in st.session_state.messages)

    with st.chat_message("assistant"):
        stream = client_deepseek.chat.completions.create(model=model_deepseek, messages=conversation, stream=True)
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})