import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df):   
    #Elimina duplicados y filas con valores nulos.
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def preprocess_features(df, scaler=None, fit_scaler=True):
    #Definir las columnas que usaremos para el modelo.
    features = [
        'duration', 'start_lat', 'start_lon', 'end_lat', 'end_lon',
        'trip_route_category', 'day_of_week', 'hour_of_day', 'round_trip'
    ]
    #Verificar que todas las columnas requeridas estén en el DataFrame.
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise KeyError(f"Faltan las siguientes columnas en el DataFrame: {missing_features}")
    
    X = df[features].copy()

    for col in ['trip_route_category', 'round_trip']:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Se requiere un scaler para transformar los datos.")
        X_scaled = scaler.transform(X)

    return X_scaled, scaler

def encode_labels(df, label_encoder=None, fit_encoder=True):
    #Codifica la columna 'passholder_type' a números usando LabelEncoder.
    if 'passholder_type' not in df.columns:
        raise KeyError("La columna 'passholder_type' no se encuentra en el DataFrame.")
    
    if fit_encoder:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['passholder_type'])
    else:
        if label_encoder is None:
            raise ValueError("Se requiere un label_encoder para transformar las etiquetas.")
        y = label_encoder.transform(df['passholder_type'])
    
    return y, label_encoder

def load_and_process_train_data(filepath):
    #Carga y procesa los datos de entrenamiento.
    column_types = {
        'start_time': str,
        'end_time': str,
        'duration': float,
        'start_lat': float,
        'start_lon': float,
        'end_lat': float,
        'end_lon': float,
        'plan_duration': float,
        'trip_route_category': str,
        'passholder_type': str,
        'start_station': str,
        'end_station': str,
        'day_of_week': int,
        'hour_of_day': int,
        'round_trip': int
    }

    df = pd.read_csv(filepath, dtype=column_types)
    df_clean = clean_data(df)

    # Asegurando el formato correcto de las fechas
    df_clean["start_time"] = pd.to_datetime(df_clean["start_time"], errors="coerce", format='%Y-%m-%d %H:%M:%S')
    df_clean["end_time"] = pd.to_datetime(df_clean["end_time"], errors="coerce", format='%Y-%m-%d %H:%M:%S')

    df_clean["duration"] = (df_clean["end_time"] - df_clean["start_time"]).dt.total_seconds() / 60
    X_scaled, scaler = preprocess_features(df_clean)
    y_encoded, label_encoder = encode_labels(df_clean)
    return X_scaled, y_encoded, scaler, label_encoder

def load_and_process_test_data(filepath, scaler):
    column_types = {
        'start_time': str,
        'end_time': str,
        'duration': float,
        'start_lat': float,
        'start_lon': float,
        'end_lat': float,
        'end_lon': float,
        'plan_duration': float,
        'trip_route_category': str,
        'start_station': str,
        'end_station': str,
        'day_of_week': int,
        'hour_of_day': int,
        'round_trip': int,
        'trip_id': str 
    }


    df = pd.read_csv(filepath, dtype=column_types)
    
    # Limpiar posibles espacios en los nombres de columnas
    df.columns = df.columns.str.strip()

    # Verificar si la columna 'trip_id' existe
    if 'trip_id' not in df.columns:
        raise KeyError("La columna 'trip_id' no se encuentra en el DataFrame de prueba.")
    test_ids = df['trip_id']
    
    # Imputar valores faltantes para 'plan_duration' 
    if "plan_duration" not in df.columns:
        df["plan_duration"] = 0  
    for col in ["start_lat", "start_lon", "end_lat", "end_lon"]:
        df[col] = df[col].fillna(df[col].median())
    
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", format='%Y-%m-%d %H:%M:%S')
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce", format='%Y-%m-%d %H:%M:%S')

    # Crear nuevas características
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    df["day_of_week"] = pd.to_datetime(df["start_time"]).dt.dayofweek
    df["hour_of_day"] = pd.to_datetime(df["start_time"]).dt.hour
    df["round_trip"] = (df["start_station"] == df["end_station"]).astype(int)
    
    X_scaled, _ = preprocess_features(df, scaler=scaler, fit_scaler=False)
    
    return X_scaled, test_ids

def save_predictions(test_ids, predictions, output_path):
    submission = pd.DataFrame({
        "trip_id": test_ids,
        "passholder_type": predictions
    })
    
    output_file = os.path.join(output_path, "analytic_bikes.csv")
    submission.to_csv(output_file, index=False)
    print(f"Predicciones guardadas en {output_file}")
