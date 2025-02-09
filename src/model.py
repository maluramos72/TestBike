import os
import joblib
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "duration", "start_lat", "start_lon", "end_lat", "end_lon",
    "plan_duration", "trip_route_category", "start_station",
    "end_station", "day_of_week", "hour_of_day", "round_trip"
]

def train_and_save_model(X, y, model_path, encoder_path, scaler_path, label_encoder, scaler):
    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Guardar el modelo y los transformadores
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    print("Modelo entrenado y guardado correctamente.")

def load_model(model_path="datasets"):
    #Modelo cargado con los transformers
    model = joblib.load(os.path.join(model_path, "best_model.pkl"))
    le = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    return model, le, scaler

def predict(model, le, scaler, X):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return le.inverse_transform(y_pred)
