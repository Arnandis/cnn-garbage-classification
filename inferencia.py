import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- CONFIGURACIÓN ---
# Ajustamos a tus parámetros reales del proyecto
IMG_SIZE = (128, 128)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predecir_imagen(ruta_imagen, ruta_modelo):
    """
    Carga el modelo y realiza la predicción sobre una imagen individual.
    """
    
    # 1. Validaciones de archivos
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: No se encuentra la imagen en '{ruta_imagen}'")
        return
    
    if not os.path.exists(ruta_modelo):
        print(f"❌ Error: No se encuentra el modelo en '{ruta_modelo}'")
        return

    # 2. Cargar el modelo entrenado
    print(f"🔄 Cargando modelo desde {ruta_modelo}...")
    try:
        model = load_model(ruta_modelo)
    except Exception as e:
        print(f"❌ Error crítico cargando el modelo: {e}")
        return

    # 3. Preprocesamiento de la imagen (Vital: igual que en el entrenamiento)
    try:
        # Cargar y redimensionar a 128x128
        img = load_img(ruta_imagen, target_size=IMG_SIZE)
        
        # Convertir a array numpy
        img_array = img_to_array(img)
        
        # Normalizar (dividir por 255) -> Esto es lo que hicimos con ImageDataGenerator(rescale=1./255)
        img_array = img_array / 255.0
        
        # Añadir la dimensión del batch: de (128, 128, 3) a (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        print(f"❌ Error procesando la imagen: {e}")
        return

    # 4. Predicción
    print("🧠 Analizando imagen...")
    prediccion = model.predict(img_array, verbose=0)
    
    # Obtener el índice de la clase con mayor probabilidad
    idx_ganador = np.argmax(prediccion)
    clase_ganadora = CLASS_NAMES[idx_ganador]
    confianza = np.max(prediccion) * 100

    # 5. Mostrar resultados
    print("\n" + "="*40)
    print(f"RESULTADO FINAL:  {clase_ganadora.upper()}")
    print(f"CONFIANZA:        {confianza:.2f}%")
    print("="*40 + "\n")
    
    print("Desglose de probabilidades:")
    for i, label in enumerate(CLASS_NAMES):
        # Dibujar una barrita visual para la probabilidad
        barra = "█" * int(prediccion[0][i] * 20) 
        print(f"  {label.ljust(10)}: {prediccion[0][i]*100:6.2f}%  {barra}")

if __name__ == "__main__":
    # Configuración de argumentos para ejecutar por consola
    parser = argparse.ArgumentParser(description='Script de Inferencia - Clasificador de Basura')
    
    # Argumento obligatorio: la imagen
    parser.add_argument('--imagen', type=str, required=True, 
                        help='Ruta del archivo de imagen a clasificar')
    
    # Argumento opcional: el modelo (por defecto busca el mejor de la fase 3)
    # CAMBIA 'mejor_modelo_fase3.keras' si tu archivo se llama diferente
    parser.add_argument('--modelo', type=str, default='mejor_modelo_fase3.keras', 
                        help='Ruta del archivo .keras del modelo')

    args = parser.parse_args()
    
    # Ejecutar la función principal
    predecir_imagen(args.imagen, args.modelo)
