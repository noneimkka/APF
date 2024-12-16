import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import ImprovedTimeSeriesPredictor
import tensorflow as tf
import os

def load_and_prepare_test_data(data_path, sequence_length=24):
    """
    Загрузка и подготовка тестовых данных
    
    Args:
        data_path: путь к данным. Может быть путём к директории с Original Data
                  или к файлу Measurement_summary_limited.csv
    """
    # Создаем словарь с диапазонами нормальных значений для каждого параметра
    normal_ranges = {
        'SO2': (0, 0.05),    # ppm
        'NO2': (0, 0.06),    # ppm
        'CO': (0, 9.0),      # ppm
        'O3': (0, 0.09),     # ppm
        'PM10': (0, 80.0),   # μg/m³
        'PM2.5': (0, 35.0)   # μg/m³
    }
    
    print("Загрузка данных...")
    
    # Определяем тип источника данных
    if os.path.isdir(data_path):
        print("Загрузка данных из Original Data...")
        
        # Загружаем информацию о станциях
        stations_info = pd.read_csv(os.path.join(data_path, "Measurement_station_info.csv"))
        print(f"\nДоступно станций: {len(stations_info)}")
        print("Список станций:")
        for _, station in stations_info.iterrows():
            print(f"  {station['Station code']}: {station['Station name(district)']} ({station['Latitude']}, {station['Longitude']})")
        
        # Загружаем информацию о параметрах
        items_info = pd.read_csv(os.path.join(data_path, "Measurement_item_info.csv"))
        print("\nПараметры измерений:")
        for _, item in items_info.iterrows():
            print(f"  {item['Item code']}: {item['Item name']} ({item['Unit of measurement']})")
            print(f"    Диапазоны: Good ≤{item['Good(Blue)']}, Normal ≤{item['Normal(Green)']}, Bad ≤{item['Bad(Yellow)']}, Very bad ≤{item['Very bad(Red)']}")
        
        # Создаем словари для маппинга
        item_code_to_name = {
            1: 'SO2',
            3: 'NO2',
            5: 'CO',
            6: 'O3',
            8: 'PM10',
            9: 'PM2.5'
        }
        
        # Загружаем основные данные
        measurements = pd.read_csv(os.path.join(data_path, "Measurement_info.csv"))
        
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
        measurements = measurements[measurements['Instrument status'] == 0]
        print(f"\nПосле фильтрации осталось {len(measurements)} измерений " +
              f"({(len(measurements) / total_measurements) * 100:.2f}% от исходных данных)")
        
        # Преобразуем даты
        measurements['Measurement date'] = pd.to_datetime(measurements['Measurement date'])
        
        # Преобразуем данные из длинного формата в широкий
        measurements['Item name'] = measurements['Item code'].map(item_code_to_name)
        data = measurements.pivot_table(
            index=['Measurement date', 'Station code'],
            columns='Item name',
            values='Average value',
            aggfunc='first'
        ).reset_index()
        
        # Добавляем информацию о станциях
        data = data.merge(stations_info, on='Station code', how='left')
        
    else:
        print("Загрузка данных из сводного файла...")
        data = pd.read_csv(data_path)
        print("\nВНИМАНИЕ: В сводном файле отсутствует информация о статусе измерений!")
        print("Все измерения будут считаться валидными.")
        
    data['Measurement date'] = pd.to_datetime(data['Measurement date'])
    
    # Проверяем наличие всех необходимых столбцов
    features = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"В данных отсутствуют следующие параметры: {missing_features}")
    
    # Добавляем временные признаки
    data['hour'] = data['Measurement date'].dt.hour
    data['day_of_week'] = data['Measurement date'].dt.dayofweek
    data['month'] = data['Measurement date'].dt.month
    data['day_of_month'] = data['Measurement date'].dt.day
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Добавляем циклические признаки
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # Нормализуем значения и анализируем выбросы
    print("\nАнализ значений параметров:")
    for feature, (min_val, max_val) in normal_ranges.items():
        original_values = data[feature].copy()
        data[feature] = data[feature].clip(lower=0)
        data[f'{feature}_normalized'] = data[feature] / max_val
        
        # Статистика по значениям
        stats = data[feature].describe()
        outliers = (original_values > max_val).sum()
        zeros = (original_values == 0).sum()
        
        print(f"\n{feature}:")
        print(f"  Диапазон нормы: 0 - {max_val}")
        print(f"  Минимум: {stats['min']:.4f}")
        print(f"  Максимум: {stats['max']:.4f}")
        print(f"  Среднее: {stats['mean']:.4f}")
        print(f"  Медиана: {stats['50%']:.4f}")
        print(f"  Стд. отклонение: {stats['std']:.4f}")
        print(f"  Количество выбросов: {outliers} ({outliers/len(data)*100:.2f}%)")
        print(f"  Количество нулевых значений: {zeros} ({zeros/len(data)*100:.2f}%)")
    
    # Сортируем данные
    data = data.sort_values(['Station code', 'Measurement date'])
    
    print("\nИтоговая статистика:")
    print(f"Количество станций: {data['Station code'].nunique()}")
    print(f"Временной диапазон: с {data['Measurement date'].min()} по {data['Measurement date'].max()}")
    print(f"Количество записей: {len(data)}")
    
    return data

def plot_predictions(true_values, pred_values, dates, station_code, feature, station_name=None, output_dir='predictions'):
    """
    Построение графика сравнения предсказанных и реальных значений
    """
    plt.figure(figsize=(15, 6))
    plt.plot(dates, true_values, label='Реальные значения', alpha=0.7)
    plt.plot(dates, pred_values, label='Предсказания', alpha=0.7)
    
    title = f'Станция {station_code}'
    if station_name:
        title += f' ({station_name})'
    title += f' - {feature}'
    
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    
    # Добавляем диапазон нормы
    normal_ranges = {
        'SO2': 0.05,
        'NO2': 0.06,
        'CO': 9.0,
        'O3': 0.09,
        'PM10': 80.0,
        'PM2.5': 35.0
    }
    if feature in normal_ranges:
        plt.axhline(y=normal_ranges[feature], color='r', linestyle='--', alpha=0.5, label='Верхняя граница нормы')
    
    # Поворачиваем метки дат для лучшей читаемости
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/prediction_{station_code}_{feature}.png')
    plt.close()

def prepare_sequences(data, feature, sequence_length=24):
    """
    Подготовка последовательностей для предсказания с гибкой обработкой пропусков
    """
    sequences = []
    dates = []
    true_values = []
    station_codes = []
    time_features = []
    
    # Проверяем наличие нормализованных значений
    normalized_feature = f'{feature}_normalized'
    if normalized_feature not in data.columns:
        raise ValueError(f"Отсутствует нормализованный столбец {normalized_feature}")
    
    print(f"\nПодготовка данных для {feature}:")
    print(f"Исходное количество записей: {len(data)}")
    
    for station in data['Station code'].unique():
        station_data = data[data['Station code'] == station].copy()
        if len(station_data) <= sequence_length:
            print(f"Пропускаем станцию {station}: недостаточно данных")
            continue
        
        # Сортируем данные по времени
        station_data = station_data.sort_values('Measurement date')
        
        # Находим и анализируем пропуски
        time_diffs = station_data['Measurement date'].diff()
        max_gap = time_diffs.max()
        avg_gap = time_diffs.mean()
        print(f"\nСтанция {station}:")
        print(f"  Максимальный пропуск: {max_gap}")
        print(f"  Средний интервал: {avg_gap}")
        
        # Находим непрерывные сегменты с разными порогами в зависимости от параметра
        if feature == 'SO2':
            # Для SO2 используем более мягкие критерии
            gap_threshold = pd.Timedelta(hours=6)  # допускаем пропуски до 6 часов
            min_segment_length = sequence_length + 1  # минимальная длина сегмента
        else:
            # Для остальных параметров используем стандартные критерии
            gap_threshold = pd.Timedelta(hours=3)  # допускаем пропуски до 3 часов
            min_segment_length = sequence_length + 1
        
        # Разбиваем на сегменты
        gaps = time_diffs > gap_threshold
        segment_ids = gaps.cumsum()
        
        total_sequences = 0
        valid_segments = 0
        
        # Обрабатываем каждый сегмент
        for segment_id in segment_ids.unique():
            segment = station_data[segment_ids == segment_id]
            
            if len(segment) < min_segment_length:
                continue
                
            valid_segments += 1
            
            # Проверяем наличие пропусков в значениях
            if segment[feature].isna().any():
                continue
            
            # Создаем скользящее окно для последовательностей
            values = segment[normalized_feature].values
            dates_array = segment['Measurement date'].values
            
            for i in range(len(values) - sequence_length):
                # Проверяем временной ряд в окне
                window_dates = dates_array[i:i + sequence_length + 1]
                time_diffs = np.diff(window_dates.astype(np.int64)) / 1e9 / 3600  # разница в часах
                
                # Проверяем максимальный пропуск в окне
                max_window_gap = np.max(time_diffs)
                if max_window_gap > gap_threshold.total_seconds() / 3600:
                    continue
                
                # Проверяем общую длительность окна
                window_duration = (window_dates[-1].astype(np.int64) - window_dates[0].astype(np.int64)) / 1e9 / 3600
                if window_duration > sequence_length * 1.5:  # допускаем растяжение окна не более чем в 1.5 раза
                    continue
                
                # Входная последовательность
                sequence = values[i:i + sequence_length]
                
                # Проверяем качество последовательности
                if np.isnan(sequence).any():
                    continue
                
                # Целевое значение
                target_value = segment[feature].iloc[i + sequence_length]
                if np.isnan(target_value):
                    continue
                
                sequences.append(sequence)
                dates.append(dates_array[i + sequence_length])
                true_values.append(target_value)
                station_codes.append(station)
                
                # Временные признаки
                target_row = segment.iloc[i + sequence_length]
                time_features.append([
                    target_row['hour_sin'],
                    target_row['hour_cos'],
                    target_row['month_sin'],
                    target_row['month_cos'],
                    target_row['is_weekend'],
                    target_row['day_of_week'],
                    target_row['day_of_month'],
                    target_row['month']
                ])
                
                total_sequences += 1
        
        if total_sequences > 0:
            print(f"  Найдено сегментов: {valid_segments}")
            print(f"  Подготовлено последовательностей: {total_sequences}")
    
    if not sequences:
        raise ValueError(f"Не удалось подготовить ни одной последовательности для {feature}")
    
    # Преобразуем списки в массивы numpy
    X = np.array(sequences).reshape(-1, sequence_length, 1)
    station_codes = np.array(station_codes)
    time_features = np.array(time_features)
    true_values = np.array(true_values)
    
    print(f"\nИтоговая статистика для {feature}:")
    print(f"Подготовлено последовательностей: {len(sequences)}")
    print(f"Количество уникальных станций: {len(np.unique(station_codes))}")
    print(f"Проверка на NaN:")
    print(f"  X: {np.isnan(X).any()}")
    print(f"  true_values: {np.isnan(true_values).any()}")
    print(f"  time_features: {np.isnan(time_features).any()}")
    print(f"Диапазоны значений:")
    print(f"  X: [{np.min(X):.4f}, {np.max(X):.4f}]")
    print(f"  true_values: [{np.min(true_values):.4f}, {np.max(true_values):.4f}]")
    
    return X, station_codes, time_features, true_values, dates

def main():
    # Создаем директорию для сохранения результатов
    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    # Загружаем тестовые данные
    print("Загрузка тестовых данных...")
    try:
        # Сначала пробуем загрузить данные из Original Data
        test_data = load_and_prepare_test_data('data/AirPollutionSeoul/Original Data')
        print("Используются данные из Original Data")
        use_station_names = True
    except Exception as e:
        print(f"Ошибка при загрузке данных из Original Data: {e}")
        print("Попытка использовать сводный файл...")
        test_data = load_and_prepare_test_data('data/Measurement_summary_limited.csv')
        print("Используется сводный файл данных")
        use_station_names = False
    
    # Создаем предиктор
    print("Загрузка моделей...")
    predictor = ImprovedTimeSeriesPredictor(sequence_length=24)
    
    # Подготавливаем кодировщик станций
    predictor.station_encoder.fit(sorted(test_data['Station code'].unique()))
    
    # Загружаем сохраненные модели
    for feature in predictor.features:
        model_path = f'models/best_model_{feature}.keras'
        if os.path.exists(model_path):
            predictor.models[feature] = tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Модель для {feature} не найдена: {model_path}")
    
    # Анализируем результаты для каждой станции и каждой характеристики
    features = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    stations = test_data['Station code'].unique()
    
    # Создаем файл для сохранения общей статистики
    with open('predictions/summary_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("Сводная статистика предсказаний\n")
        f.write("=" * 50 + "\n\n")
        
        for feature in features:
            f.write(f"\nАнализ {feature}:\n")
            print(f"\nАнализ {feature}:")
            
            # Подготавливаем данные для текущего параметра
            X, station_codes, time_features, true_values, dates = prepare_sequences(
                test_data, feature
            )
            
            # Нормализуем true_values для сравнения с предсказаниями
            normal_ranges = {
                'SO2': 0.05,
                'NO2': 0.06,
                'CO': 9.0,
                'O3': 0.09,
                'PM10': 80.0,
                'PM2.5': 35.0
            }
            
            # Получаем предсказания (они будут в нормализованном виде)
            predictions_norm = predictor.models[feature].predict({
                'time_series_input': X,
                'station_input': predictor.station_encoder.transform(station_codes),
                'time_features_input': time_features
            }).flatten()
            
            # Денормализуем предсказания
            predictions = predictions_norm * normal_ranges[feature]
            
            # Анализируем результаты по станциям
            feature_mae = []
            feature_rmse = []
            feature_mape = []
            
            for station in stations:
                station_mask = station_codes == station
                if not any(station_mask):
                    continue
                
                station_true = true_values[station_mask]
                station_pred = predictions[station_mask]
                station_dates = [dates[i] for i, mask in enumerate(station_mask) if mask]
                
                # Проверяем данные
                if len(station_true) == 0 or len(station_pred) == 0:
                    print(f"Пропускаем станцию {station}: нет данных")
                    continue
                
                # Получаем название станции, если доступно
                station_name = None
                if use_station_names:
                    station_data = test_data[test_data['Station code'] == station].iloc[0]
                    station_name = station_data['Station name(district)']
                
                # Метрики
                mae = np.mean(np.abs(station_true - station_pred))
                rmse = np.sqrt(np.mean((station_true - station_pred) ** 2))
                
                # Избегаем деления на ноль при расчете MAPE
                nonzero_mask = station_true != 0
                if any(nonzero_mask):
                    mape = np.mean(np.abs((station_true[nonzero_mask] - station_pred[nonzero_mask]) / 
                                        station_true[nonzero_mask])) * 100
                else:
                    mape = np.nan
                
                feature_mae.append(mae)
                feature_rmse.append(rmse)
                if not np.isnan(mape):
                    feature_mape.append(mape)
                
                # Записываем статистику в файл
                f.write(f"\nСтанция {station}")
                if station_name:
                    f.write(f" ({station_name})")
                f.write(":\n")
                f.write(f"  MAE: {mae:.4f}\n")
                f.write(f"  RMSE: {rmse:.4f}\n")
                f.write(f"  MAPE: {mape:.2f}%\n")
                f.write(f"  Диапазон реальных значений: {np.min(station_true):.4f} - {np.max(station_true):.4f}\n")
                f.write(f"  Диапазон предсказаний: {np.min(station_pred):.4f} - {np.max(station_pred):.4f}\n")
                
                # Выводим статистику в консоль
                print(f"\nСтанция {station}" + (f" ({station_name})" if station_name else "") + ":")
                print(f"  MAE: {mae:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  Диапазон реальных значений: {np.min(station_true):.4f} - {np.max(station_true):.4f}")
                print(f"  Диапазон предсказаний: {np.min(station_pred):.4f} - {np.max(station_pred):.4f}")
                
                # Строим график
                plot_predictions(station_true, station_pred, station_dates, station, feature, station_name)
            
            if feature_mae and feature_rmse:
                # Записываем среднюю статистику по параметру
                f.write(f"\nСредние значения для {feature}:\n")
                f.write(f"  MAE: {np.mean(feature_mae):.4f} ± {np.std(feature_mae):.4f}\n")
                f.write(f"  RMSE: {np.mean(feature_rmse):.4f} ± {np.std(feature_rmse):.4f}\n")
                if feature_mape:
                    f.write(f"  MAPE: {np.mean(feature_mape):.2f}% ± {np.std(feature_mape):.2f}%\n")
                else:
                    f.write("  MAPE: не удалось рассчитать (деление на ноль)\n")
            else:
                f.write(f"\nНе удалось рассчитать статистику для {feature}: недостаточно данных\n")
            
            f.write("-" * 50 + "\n")
    
    print("\nГотово! Результаты сохранены в директории 'predictions/'")
    print("Подробная статистика доступна в файле 'predictions/summary_statistics.txt'")

if __name__ == "__main__":
    main() 