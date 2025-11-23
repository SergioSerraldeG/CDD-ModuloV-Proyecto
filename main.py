import streamlit as st
import os

# 💰 Emoji de finanzas añadido al título
st.title("💰 CDD-ModuloV-Proyectos")
st.caption("Inversiones Simplificadas")

# Chat input
prompt = st.chat_input("¿En qué te puedo ayudar?")

if prompt:
    st.write(f"El usuario ha enviado el siguiente mensaje: {prompt}")
