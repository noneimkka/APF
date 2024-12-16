import os
import pandas as pd
import numpy as np
from model import ImprovedTimeSeriesPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
import glob

def load_and_prepare_data(data_dir):
    """
    Загрузка и подготовка данных из нескольких файлов
    
    Статусы измерительных приборов:
    0: Нормальное состояние
    1: Требуется калибровка
    2: Аномальное состояние
    4: Отключение питания
    8: В ремонте
    9: Аномальные данные
    """
    print("Загрузка информации о параметрах...")
    items_info = pd.read_csv(os.path.join(data_dir, "Measurement_item_info.csv"))
    
    # Создаем словари для маппинга
    item_code_to_name = {
        1: 'SO2',
        3: 'NO2',
        5: 'CO',
        6: 'O3',
        8: 'PM10',
        9: 'PM2.5'
    }
    
    # Создаем словарь с диапазонами нормальных значений
    normal_ranges = {
        'SO2': (0, 0.05),    # Good to Normal range
        'NO2': (0, 0.06),
        'CO': (0, 9.0),
        'O3': (0, 0.09),
        'PM10': (0, 80.0),
        'PM2.5': (0, 35.0)
    }
    
    print("Загрузка основных данных...")
    measurements = pd.read_csv(os.path.join(data_dir, "Measurement_info.csv"))
    
    # Анализ статусов измерений до фильтрации
    status_counts = measurements['Instrument status'].value_counts()
    total_measurements = len(measurements)
    
    print("\nСтатистика по статусам измерений до фильтрации:")
    for status, count in status_counts.items():
        percentage = (count / total_measurements) * 100
        status_description = {
            0: "Нормальное состояние",
            1: "Требуется калибровка",
            2: "Аномальное состояние",
            4: "Отключение питания",
            8: "В ремонте",
            9: "Аномальные данные"
        }.get(status, "Неизвестный статус")
        print(f"Статус {status} ({status_description}): {count} измерений ({percentage:.2f}%)")
    
    # Фильтрация данных по статусу
    # Оставляем только нормальные измерения и те, что требуют калибровки
    valid_statuses = [0]  # Можно добавить 1, если хотим включить данные, требующие калибровки
    measurements = measurements[measurements['Instrument status'].isin(valid_statuses)]
    
    print(f"\nПосле фильтрации осталось {len(measurements)} измерений " +
          f"({(len(measurements) / total_measurements) * 100:.2f}% от исходных данных)")
    
    # Преобразуем даты
    measurements['Measurement date'] = pd.to_datetime(measurements['Measurement date'])
    
    # Преобразуем данные из длинного формата в широкий
    measurements['Item name'] = measurements['Item code'].map(item_code_to_name)
    
    # Разворачиваем данные так, чтобы каждый параметр был в отдельном столбце
    data = measurements.pivot_table(
        index=['Measurement date', 'Station code', 'Instrument status'],  # Добавляем статус в индекс
        columns='Item name',
        values='Average value',
        aggfunc='first'
    ).reset_index()
    
    # Проверяем наличие всех необходимых столбцов
    required_columns = ['Measurement date', 'Station code', 'SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Внимание: отсутствуют следующие столбцы: {missing_columns}")
    
    # Удаление строк с пропущенными значениями
    initial_rows = len(data)
    data = data.dropna(subset=['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5'])
    dropped_rows = initial_rows - len(data)
    if dropped_rows > 0:
        print(f"\nУдалено {dropped_rows} строк с пропущенными значениями " +
              f"({(dropped_rows / initial_rows) * 100:.2f}% от данных после фильтрации)")
    
    # Добавляем временные признаки
    data['hour'] = data['Measurement date'].dt.hour
    data['day_of_week'] = data['Measurement date'].dt.dayofweek
    data['month'] = data['Measurement date'].dt.month
    data['day_of_month'] = data['Measurement date'].dt.day
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Добавляем циклические признаки для времени
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # Нормализуем значения относительно нормальных диапазонов
    for feature, (min_val, max_val) in normal_ranges.items():
        # Обрезаем экстремальные значения
        data[feature] = data[feature].clip(lower=0)
        data[f'{feature}_normalized'] = data[feature] / max_val
        
        # Добавляем признак выхода за пределы нормы
        data[f'{feature}_out_of_range'] = (data[feature] > max_val).astype(int)
    
    # Сортировка по дате и станции
    data = data.sort_values(['Station code', 'Measurement date'])
    
    print("\nИтоговая статистика по данным:")
    print(f"Количество станций: {data['Station code'].nunique()}")
    print(f"Временной диапазон: с {data['Measurement date'].min()} по {data['Measurement date'].max()}")
    print(f"Количество записей: {len(data)}")
    print("\nСтатистика по параметрам:")
    
    for feature in ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']:
        print(f"\n{feature}:")
        print("Оригинальные значения:")
        print(data[feature].describe())
        print("\nНормализованные значения:")
        print(data[f'{feature}_normalized'].describe())
        print(f"Процент выхода за пределы нормы: {data[f'{feature}_out_of_range'].mean() * 100:.2f}%")
    
    return data

def plot_training_history(histories):
    """
    Построение графиков обучения для каждого параметра
    """
    metrics = ['loss', 'mae', 'rmse']
    titles = ['Model Loss', 'Model MAE', 'Model RMSE']
    
    for feature, history in histories.items():
        plt.figure(figsize=(20, 5))
        for i, (metric, title) in enumerate(zip(metrics, titles), 1):
            plt.subplot(1, 3, i)
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{feature} - {title}')
            plt.xlabel('Epoch')
            plt.ylabel(title)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'models/training_history_{feature}.png')
        plt.close()

def main():
    # Создание директорий
    for directory in ['models', 'logs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Загрузка данных
    print("Загрузка и подготовка данных...")
    try:
        # Сначала пробуем загрузить расширенный набор данных
        data = load_and_prepare_data('data/AirPollutionSeoul/Original Data')
        print("Используется расширенный набор данных")
    except Exception as e:
        print(f"Ошибка при загрузке расширенного набора данных: {e}")
        print("Используется ограниченный набор данных")
        data = load_and_prepare_data('data')
    
    print(f"Уникальные станции в исходных данных: {sorted(data['Station code'].unique())}")
    print(f"Временной диапазон: с {data['Measurement date'].min()} по {data['Measurement date'].max()}")
    print(f"Общее количество записей: {len(data)}")
    
    # Создание и обучение модели
    print("Подготовка модели...")
    predictor = ImprovedTimeSeriesPredictor(sequence_length=24)
    
    # Сначала подготовим кодировщик станций на всех данных
    predictor.station_encoder.fit(sorted(data['Station code'].unique()))
    print(f"Маппинг станций: {dict(zip(predictor.station_encoder.classes_, predictor.station_encoder.transform(predictor.station_encoder.classes_)))}")
    
    # Подготавливаем данные для всех станций сразу
    print("Подготовка данных...")
    X, station_ids, y, time_features_data = predictor.prepare_data(data)
    
    print("\nИтоговые размерности данных:")
    for feature in predictor.features:
        print(f"{feature}:")
        print(f"  X: {X[feature].shape}")
        print(f"  y: {y[feature].shape}")
    print(f"station_ids: {station_ids.shape}")
    print(f"time_features: {time_features_data.shape}")
    
    # Обучение модели
    print("Начало обучения...")
    histories = predictor.train(
        X, 
        station_ids, 
        y,
        time_features_data,
        epochs=30,
        batch_size=64,
        validation_split=0.2
    )
    
    # Построение графиков обучения
    print("Построение графиков обучения...")
    plot_training_history(histories)
    
    print("Обучение завершено. Модели и графики сохранены в директории models/")

if __name__ == "__main__":
    main() 