import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.parameters = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
        self.normal_ranges = {
            'SO2': 0.05,    # ppm
            'NO2': 0.06,    # ppm
            'CO': 9.0,      # ppm
            'O3': 0.09,     # ppm
            'PM10': 80.0,   # μg/m³
            'PM2.5': 35.0   # μg/m³
        }
        
        # Инициализируем кодировщик станций
        self.station_encoder = LabelEncoder()
        
        # Загружаем информацию о станциях для инициализации кодировщика
        stations_info = pd.read_csv('data/AirPollutionSeoul/Original Data/Measurement_station_info.csv')
        self.station_encoder.fit(sorted(stations_info['Station code'].unique()))
        
        # Загружаем модели
        models_dir = 'models'
        for param in self.parameters:
            model_path = os.path.join(models_dir, f'best_model_{param}.keras')
            
            if os.path.exists(model_path):
                try:
                    self.models[param] = tf.keras.models.load_model(model_path)
                except Exception as e:
                    print(f"Ошибка загрузки модели {param}: {str(e)}")
    
    def prepare_features(self, station_code, target_datetime):
        """Подготовка признаков для предсказания"""
        time_series = tf.zeros((1, 24, 1), dtype=tf.float32)
        
        station_encoded = self.station_encoder.transform([station_code])
        station = tf.convert_to_tensor(station_encoded.reshape(-1, 1), dtype=tf.int32)
        
        hour_sin = np.sin(2 * np.pi * target_datetime.hour / 24)
        hour_cos = np.cos(2 * np.pi * target_datetime.hour / 24)
        month_sin = np.sin(2 * np.pi * target_datetime.month / 12)
        month_cos = np.cos(2 * np.pi * target_datetime.month / 12)
        
        time_features = tf.convert_to_tensor([[
            hour_sin,
            hour_cos,
            month_sin,
            month_cos,
            float(target_datetime.weekday() in [5, 6]),
            float(target_datetime.weekday()),
            float(target_datetime.day),
            float(target_datetime.month)
        ]], dtype=tf.float32)
        
        return {
            'time_series_input': time_series,
            'station_input': station,
            'time_features_input': time_features
        }
    
    def predict_range(self, station_code, target_datetime):
        """Предсказание значений для временного диапазона ±2 часа"""
        predictions = {param: [] for param in self.parameters}
        time_points = []
        
        for hour_offset in range(-2, 3):
            current_datetime = target_datetime + timedelta(hours=hour_offset)
            time_points.append(current_datetime)
            
            features = self.prepare_features(station_code, current_datetime)
            
            for param in self.parameters:
                if param in self.models:
                    try:
                        pred = self.models[param](features, training=False)
                        pred_value = float(pred.numpy()[0][0])
                        pred_value = pred_value * self.normal_ranges[param]
                        predictions[param].append(pred_value)
                    except Exception as e:
                        print(f"Ошибка предсказания для {param}: {str(e)}")
                        predictions[param].append(None)
                else:
                    predictions[param].append(None)
        
        return predictions, time_points
    
    def get_available_parameters(self):
        """Возвращает список параметров, для которых есть модели"""
        return [param for param in self.parameters if param in self.models] 