import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from prompts import stronger_prompt

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
model_openai = "gpt-4o-mini"

st.title("📊 PepMex")
st.caption("Experto en PepsiCo México")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "¿En qué te puedo ayudar?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Escribe tu mensaje aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    conversation = [{"role": "system", "content": stronger_prompt}]
    conversation.extend(st.session_state.messages)

    with st.chat_message("assistant"):
        stream = client_openai.chat.completions.create(
            model=model_openai,
            messages=conversation,
            stream=True
        )
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )