# TestBikes

Este es un proyecto de predicción de tipos de abonados de bicicletas utilizando datos de viaje. 
El modelo predice el tipo de abonado a partir de varias características de los viajes de bicicletas.

## Requisitos

- Python 3.12.6 o superior
- Bibliotecas necesarias:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - sklearn
  - xgboost
  - lightgbm
  - joblib

## Estructura del Proyecto

TestBikes.git/
│── datasets/
│   ├── test.csv
│   ├── train.csv
│   ├── test_clean.csv
│   ├── train_clean.csv
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│── src/
│   ├── main.py
│   ├── data_processing.py
│   ├── model.py
│   ├── predict.py
│   ├── TestBikes.ipynb
│── docs/
│   ├── Proyecto_TestBikes.pdf
│── diagrama/
│   ├── Diagrama Pipeline.pdf
│── README.md
│── requiments.txt
│── Dockerfile
│── .dockerignore


- **main.py**: Archivo principal que ejecuta el flujo del proyecto.
- **data_processing.py**: Contiene funciones para limpiar, procesar y preprocesar los datos.
- **model.py**: Define las características que el modelo utiliza para entrenar y realizar predicciones.
- **predict.py**: Carga el modelo entrenado y realiza predicciones sobre el conjunto de datos de prueba.
- **TestBikes.IPYNB**: Archivo Principal del codigo completo -realizado en google colabs (python)- donde crea los archivos preentrenados- (bets_model, scaler, label_encoder) asi como tambien los archivos test_clean and train_clean
- **datsets/**: Carpeta con los archivos de datos y modelos preentrenados. (No Compartidos en el github por el tamaño de los archivos).
- **docs/**: Carpeta con el archivo proyecto_testBikes.pdf (archivo de presentación del proyecto)
- **diagrama/**: Carpeta con el archivo diagrama pipeline.pdf (diagrama del proyecto)
- **requiments.txt**: Archivo de librerias requeridas
- **Dockerfile**: Archivo docker

## Primeros pasos

- **Clonar el Repositorio**: git clone https://github.com/maluramos72/TestBikes.git
- **Construye la imagen de Docker**: bash docker build -t testbikes .
- **Construye la imagen de Docker**: bash docker run --rm -it testbikes

## Cómo Usar

NOTA. Asegúrate de tener todos los archivos necesarios:
   - `test_set.csv`: Conjunto de datos de prueba.
   - `train_set.csv`: Conjunto de datos de entrenamiento.
     
## Se tienen dos opciones.
1. Como opción recomendada primero ejecutar TestBikes.ipynb en google colabs o jupiter
       * Esto genera los archivos preentrenados
       * Y el archivo "analytics-bike.csv" con las predicciones.

2. Como opción ejecutar desde VisualStudio.
    a). Asegúrate de tener todos los archivos necesarios:
         - `test_clean.csv`: Conjunto de datos de prueba clean.
         - `train_clean.csv`: Conjunto de datos de entrenamiento clean.
         - `best_model.pkl`: Modelo previamente entrenado.
         - `scaler.pkl`: Escalador para normalizar las características.
         - `label_encoder.pkl`: Codificador para las etiquetas de los tipos de abonados.
    b). Ejecutar el archivo `main.py` para procesar los datos y realizar las predicciones:
          *Esto generará un archivo `analytics-bike.csv` con las predicciones.

## Descarga de los Datos

Los datos necesarios para este proyecto se encuentran disponibles en Kaggle. Para descargarlos:

1. Ve al siguiente enlace: [Kaggle Dataset](https://www.kaggle.com/t/e82d8dd1223a4a459037106a2acab561).
2. Haz clic en "Download" para obtener los archivos CSV.
3. Descomprime el archivo descargado y coloca los archivos en la carpeta `datasets/` de este repositorio.

Una vez que los datos estén en la carpeta adecuada, puedes ejecutar el código para el análisis y entrenamiento del modelo.


