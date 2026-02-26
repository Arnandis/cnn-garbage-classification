UD05 Proyecto Final - Pau Arnandis Martínez
Dataset utilizado
Garbage Classification (Kaggle)

URL de descarga: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

Nota: La descarga está automatizada dentro del propio Notebook usando la API de Kaggle.

Fases completadas
[x] Fase 1 - Fundamentos

[x] Fase 2 - Experimentación

[x] Fase 3 - Excelencia

[x] Bonus: Opción de Despliegue Web interactivo (App con Streamlit)

Instrucciones de ejecución
Entorno
Python 3.x + TensorFlow 2.x
Recomendado: Google Colab o Kaggle Notebooks (debido a la necesidad de aceleración por hardware GPU para el entrenamiento).

Instalación
Para ejecutar en local o instalar dependencias, utiliza el siguiente comando:

Bash
pip install -r requirements.txt
Descarga del dataset
No es necesario descargarlo manualmente. El Notebook contiene un script de automatización en la primera celda. Solo se requiere:

Disponer de un archivo kaggle.json (API Token de Kaggle).

Ejecutar la primera celda del Notebook, que solicitará subir el archivo kaggle.json.

El dataset se descargará y descomprimirá automáticamente en la carpeta dataset_garbage/.

Ejecución
Todo el proyecto (Fases 1, 2 y 3) ha sido unificado y estructurado secuencialmente en un único cuaderno para facilitar su corrección y lectura:

Abrir y ejecutar las celdas en orden secuencial del archivo UD05_Proyecto_PauArnandis.ipynb.

Script de inferencia (Fase 3)
Se incluye un script Python para probar el modelo desde la terminal. Asegúrate de tener el modelo mejor_modelo_fase1.keras en la misma carpeta.

Bash
python inferencia.py
(El script buscará automáticamente una imagen aleatoria en la carpeta del dataset para realizar la demostración).

Despliegue Web (Bonus)
Se ha incluido una aplicación web desarrollada con Streamlit (app.py). Para arrancarla en un entorno local:

Bash
python -m streamlit run app.py
Resultados principales
Dataset: Garbage Classification

Accuracy objetivo: > 80.00%

Accuracy obtenido (Fase 1 Base): 72.11%

Mejor accuracy (Fases 2+3 - CNN_BatchNorm_Deep): 85.53%