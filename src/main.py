from data_processing import load_and_process_train_data, load_and_process_test_data, save_predictions
from model import train_and_save_model
import os
import pandas as pd
import joblib

# Definir rutas de archivos
train_data_path = os.path.join("datasets", "train_clean.csv")
test_data_path = os.path.join("datasets", "test_set.csv") 
model_path = os.path.join("datasets", "best_model.pkl")
encoder_path = os.path.join("datasets", "label_encoder.pkl")
scaler_path = os.path.join("datasets", "scaler.pkl")
output_path = "datasets"  

# Paso 1: Entrenamiento
# Cargar y procesar datos de entrenamiento
X_train, y_train, scaler, label_encoder = load_and_process_train_data(train_data_path)
# Entrenar y guardar el modelo y los transformadores
train_and_save_model(X_train, y_train, model_path, encoder_path, scaler_path, label_encoder, scaler)

# Paso 2: Predicción
# Cargar el scaler guardado para usarlo en la predicción
scaler = joblib.load(scaler_path)
# Cargar y procesar datos de prueba (utilizando el scaler guardado)
X_test, test_ids = load_and_process_test_data(test_data_path, scaler)
# Cargar el modelo entrenado y predicciones
best_model = joblib.load(model_path)
y_pred = best_model.predict(X_test)
# Convertir las predicciones numéricas a las etiquetas originales
label_encoder = joblib.load(encoder_path) 
pred_labels = label_encoder.inverse_transform(y_pred)

# Paso 3: Guardar predicciones
save_predictions(test_ids, pred_labels, output_path)
