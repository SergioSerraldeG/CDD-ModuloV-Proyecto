import os
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar la clave de API desde el archivo .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: No se pudo encontrar la GOOGLE_API_KEY en el archivo .env")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Modelos de IA generativa disponibles para tu clave:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"- {model.name}")
    except Exception as e:
        print(f"Ocurrió un error al contactar la API de Google: {e}")