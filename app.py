from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.linear_model import Ridge
import joblib
import re
import numpy as np

app = FastAPI()

class CarFeatures(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class CarFeatureBatch(BaseModel):
    entries: List[CarFeatures]

# Загрузка моделей
model_path = 'elasticnet_model.pkl'
scaler_path = 'scaler.pkl'

car_price_model = joblib.load(model_path)
scaling_transformer = joblib.load(scaler_path)

# Обработка пропусков и преобразование данных
missing_features = ['mileage', 'engine', 'max_power', 'torque', 'seats']
default_values = [19.3, 1248.0, 82.0, 171.0, 5.0]

def parse_numeric_value(input_value):
    if pd.isnull(input_value):
        return np.nan
    match = re.search(r'[\d.]+', str(input_value))
    return float(match.group(0)) if match else np.nan

def convert_torque_to_nm(torque_value):
    if pd.isnull(torque_value):
        return np.nan
    match = re.search(r'([\d.]+)\s*(Nm|kgm)', str(torque_value), re.IGNORECASE)
    if match:
        torque_magnitude = float(match.group(1))
        if match.group(2).lower() == 'kgm':
            torque_magnitude *= 9.80665
        return torque_magnitude
    return np.nan

def transform_input_data(input_data: CarFeatures) -> pd.DataFrame:
    temp_df = pd.DataFrame([input_data.dict()])

    # Преобразование числовых признаков
    temp_df['mileage'] = temp_df['mileage'].apply(parse_numeric_value)
    temp_df['engine'] = temp_df['engine'].apply(parse_numeric_value)
    temp_df['max_power'] = temp_df['max_power'].apply(parse_numeric_value)
    temp_df['torque'] = temp_df['torque'].apply(convert_torque_to_nm)

    # Заполнение пропусков
    for feature, default in zip(missing_features, default_values):
        temp_df[feature].fillna(default, inplace=True)

    # Приведение типов
    temp_df[['mileage', 'engine', 'max_power']] = temp_df[['mileage', 'engine', 'max_power']].astype(float)
    temp_df[['engine', 'seats']] = temp_df[['engine', 'seats']].astype(int)

    # Масштабирование данных
    scaled_data = scaling_transformer.transform(temp_df.select_dtypes(include=['int', 'float']))
    return scaled_data

@app.post("/single_prediction")
def make_single_prediction(car_features: CarFeatures) -> float:
    prepared_data = transform_input_data(car_features)
    predicted_price = car_price_model.predict(prepared_data)
    print('Предсказание выполнено студентом Аладинским Г.А = ' + str(float(predicted_price[0])))
    return float(predicted_price[0])

@app.post("/batch_prediction")
def process_file_and_predict(uploaded_file: UploadFile = File(...)) -> str:
    input_dataframe = pd.read_csv(uploaded_file.file)
    processed_entries = pd.concat(
        [transform_input_data(CarFeatures(**record)) for _, record in input_dataframe.iterrows()],
        axis=0
    )
    price_predictions = car_price_model.predict(processed_entries)
    print('Пакетное предсказание выполнено студентом Аладинским Г.А')
    input_dataframe['predicted_price'] = price_predictions
    output_file_name = 'predicted_prices.csv'
    input_dataframe.to_csv(output_file_name, index=False)
    return output_file_name
