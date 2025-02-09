import pandas as pd
import joblib
import os
from model import FEATURES 
from data_processing import preprocess_features, save_predictions

MODEL_PATH = "datasets/model.pkl"
SCALER_PATH = "datasets/scaler.pkl"
LABEL_ENCODER_PATH = "datasets/label_encoder.pkl"

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("Modelo, scaler y label encoder cargados correctamente.")
        return model, scaler, label_encoder
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None, None

def load_test_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Datos de prueba cargados correctamente.")
        return df
    except Exception as e:
        print(f"Error al cargar los datos de prueba: {e}")
        return None

def load_model_and_predict(test_filepath, output_path):
    model, scaler, label_encoder = load_model()
    if model is None:
        return None
    
    df_test = load_test_data(test_filepath)
    if df_test is None:
        return None

    if not all(feature in df_test.columns for feature in FEATURES):
        print("Error: Falta alguna columna requerida en el archivo de prueba.")
        return None

    X_test_scaled, _ = preprocess_features(df_test, scaler, fit_scaler=False)

    y_pred = model.predict(X_test_scaled)

    pred_labels = label_encoder.inverse_transform(y_pred)

    save_predictions(df_test['trip_id'], pred_labels, output_path)
    
    return df_test

if __name__ == "__main__":
    test_filepath = "datasets/test_clean.csv" 
    output_path = "datasets" 
    load_model_and_predict(test_filepath, output_path)


