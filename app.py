import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# --- CONFIGURACIÓN ---
# NOTA: Cuando entregues, asegúrate de que el modelo .keras esté en la misma carpeta que este archivo
MODEL_PATH = 'mejor_modelo_fase1.keras' 
CLASS_NAMES = ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plástico', 'Basura']

st.set_page_config(page_title="Clasificador de Basura", page_icon="♻️")

st.title("♻️ ¿Dónde lo tiro?")
st.write("Sube una foto y te diré qué tipo de residuo es.")

# Cargar modelo (sin mostrar error si no está, para que quede limpio)
@st.cache_resource
def load_keras_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

model = load_keras_model()

if model is None:
    st.warning(f"⚠️ No encuentro el archivo '{MODEL_PATH}'. Ponlo en la misma carpeta que este script.")

# Subir archivo
file = st.file_uploader("Sube tu imagen (JPG/PNG)", type=["jpg", "png", "jpeg"])

if file and model:
    image = Image.open(file)
    st.image(image, width=300)
    
    # Preprocesar igual que en el entrenamiento
    img_resized = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predecir
    pred = model.predict(img_array, verbose=0)
    clase = CLASS_NAMES[np.argmax(pred)]
    conf = np.max(pred) * 100
    
    st.success(f"Es: **{clase}** ({conf:.1f}%)")
