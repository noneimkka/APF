import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_station_info(file_path='data/AirPollutionSeoul/Original Data/Measurement_station_info.csv'):
    """Загрузка информации о станциях измерения"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Ошибка при загрузке информации о станциях: {str(e)}")

def format_datetime(dt):
    """Форматирование даты и времени для отображения"""
    return dt.strftime('%d.%m.%Y %H:00')

def create_time_features(dt):
    """Создание циклических временных признаков"""
    hour_sin = np.sin(2 * np.pi * dt.hour / 24)
    hour_cos = np.cos(2 * np.pi * dt.hour / 24)
    month_sin = np.sin(2 * np.pi * dt.month / 12)
    month_cos = np.cos(2 * np.pi * dt.month / 12)
    
    return {
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_weekend': float(dt.weekday() in [5, 6]),
        'weekday': float(dt.weekday()),
        'day': float(dt.day),
        'month': float(dt.month)
    }

def get_time_range(center_datetime, hours=2):
    """Получение временного диапазона для прогноза"""
    time_points = []
    for hour_offset in range(-hours, hours + 1):
        time_points.append(center_datetime + timedelta(hours=hour_offset))
    return time_points 