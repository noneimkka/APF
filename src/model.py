import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Embedding, Concatenate, 
    BatchNormalization, MultiHeadAttention, Conv1D, 
    GlobalAveragePooling1D, Add, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import time
import os

# Оптимизация CPU
tf.config.threading.set_inter_op_parallelism_threads(6)
tf.config.threading.set_intra_op_parallelism_threads(6)

# Включаем XLA оптимизацию
tf.config.optimizer.set_jit(True)

class CustomProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, feature_name):
        super(CustomProgressCallback, self).__init__()
        self.epoch_start_time = None
        self.feature_name = feature_name
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n{self.feature_name} - Эпоха {epoch + 1}/{self.params['epochs']}")
        self.seen = 0
        self.steps = self.params['steps']
        
    def on_batch_end(self, batch, logs=None):
        self.seen += 1
        if self.seen % 20 == 0 or self.seen == self.steps:
            percentage = (self.seen / self.steps) * 100
            elapsed_time = time.time() - self.epoch_start_time
            eta = (elapsed_time / self.seen) * (self.steps - self.seen)
            
            metrics_str = ' - '.join([
                f"{k}: {v:.4f}" for k, v in logs.items()
            ])
            
            print(f"\rПрогресс: {percentage:.1f}% - {metrics_str} - ETA: {eta:.0f}s", end='')
    
    def on_epoch_end(self, epoch, logs=None):
        total_time = time.time() - self.epoch_start_time
        metrics_str = ' - '.join([
            f"{k}: {v:.4f}" for k, v in logs.items()
        ])
        print(f"\rЗавершено - {metrics_str} - {total_time:.0f}s")

class ImprovedTimeSeriesPredictor:
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.scalers = {}
        self.features = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
        self.models = {}
        self.station_encoder = LabelEncoder()
        
    def create_feature_model(self, input_shape, num_stations, feature_name):
        """
        Создание улучшенной модели для одного параметра
        """
        # Входной слой для временного ряда
        time_series_input = Input(shape=input_shape, name='time_series_input')
        
        # Улучшенный embedding для станций
        station_input = Input(shape=(1,), name='station_input')
        station_embedding = Embedding(num_stations, 32)(station_input)
        station_embedding = tf.keras.layers.Flatten()(station_embedding)
        
        # Временные признаки
        time_features_input = Input(shape=(8,), name='time_features_input')  # hour_sin, hour_cos, month_sin, month_cos, is_weekend, day_of_week, day_of_month, month
        
        # Сверточный блок для извлечения локальных паттернов
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(time_series_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(conv1)
        conv1 = BatchNormalization()(conv1)
        
        conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(time_series_input)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(conv2)
        conv2 = BatchNormalization()(conv2)
        
        # Объединяем сверточные блоки
        conv_concat = Concatenate()([conv1, conv2])
        
        # Блок LSTM с механизмом внимания
        lstm1 = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))(conv_concat)
        lstm1 = LayerNormalization()(lstm1)
        
        # Механизм внимания
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(lstm1, lstm1)
        attention = Dropout(0.2)(attention)
        lstm1 = Add()([lstm1, attention])
        lstm1 = LayerNormalization()(lstm1)
        
        # Второй слой LSTM
        lstm2 = LSTM(64, kernel_regularizer=l2(0.001))(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Объединяем все признаки
        combined = Concatenate()([lstm2, station_embedding, time_features_input])
        
        # Полносвязные слои с остаточными связями
        dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.2)(dense1)
        
        dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.2)(dense2)
        
        # Остаточное соединение
        dense3 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dense2)
        dense3 = BatchNormalization()(dense3)
        dense3 = Add()([dense2, dense3])
        dense3 = Dropout(0.2)(dense3)
        
        # Выходной слой
        output = Dense(1)(dense3)
        
        model = Model(
            inputs=[time_series_input, station_input, time_features_input],
            outputs=output
        )
        
        # Используем Adam с уменьшенным learning rate и градиентным клиппингом
        optimizer = Adam(
            learning_rate=5e-4,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Более устойчивая к выбросам функция потерь
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    def prepare_data(self, data):
        """
        Подготовка данных для обучения модели
        """
        print(f"Уникальные коды станций в данных: {sorted(data['Station code'].unique())}")
        
        # Подготовка временных признаков
        time_features = [
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'is_weekend', 'day_of_week', 'day_of_month', 'month'
        ]
        
        # Нормализуем временные признаки
        time_scalers = {}
        for feature in ['day_of_week', 'day_of_month', 'month']:
            if feature not in self.scalers:
                scaler = StandardScaler()
                self.scalers[feature] = scaler
                data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        
        # Подготовка последовательностей для каждого параметра
        X, y = {}, {}
        station_ids = []
        time_features_data = []
        first_feature = True
        
        # Группируем данные по станциям и датам
        data = data.sort_values(['Station code', 'Measurement date'])
        
        # Для каждой станции создаем последовательности
        for station in data['Station code'].unique():
            station_data = data[data['Station code'] == station]
            
            if len(station_data) <= self.sequence_length:
                continue
            
            # Создаем последовательности для этой станции
            for i in range(len(station_data) - self.sequence_length):
                if first_feature:
                    station_ids.append(station)
                    # Добавляем временные признаки для целевого момента времени
                    time_features_data.append(
                        station_data[time_features].iloc[i + self.sequence_length].values
                    )
                
                # Для каждого параметра создаем отдельные последовательности
                for feature in self.features:
                    if feature not in X:
                        X[feature] = []
                        y[feature] = []
                    
                    # Используем нормализованные значения
                    feature_sequence = station_data[f'{feature}_normalized'].iloc[i:(i + self.sequence_length)].values
                    X[feature].append(feature_sequence)
                    
                    # Целевое значение - следующий час (тоже нормализованное)
                    y[feature].append(station_data[f'{feature}_normalized'].iloc[i + self.sequence_length])
            
            first_feature = False
        
        # Преобразуем коды станций
        station_ids = self.station_encoder.transform(station_ids)
        
        # Преобразуем списки в массивы numpy
        for feature in self.features:
            X[feature] = np.array(X[feature]).reshape(-1, self.sequence_length, 1)
            y[feature] = np.array(y[feature])
        
        time_features_data = np.array(time_features_data)
        
        return X, np.array(station_ids), y, time_features_data
    
    def train(self, X_dict, station_ids, y_dict, time_features_data, epochs=20, batch_size=128, validation_split=0.2):
        """
        Обучение отдельной модели для каждого параметра
        """
        histories = {}
        
        # Создаем директории для сохранения моделей и логов
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        for feature in self.features:
            print(f"\nОбучение модели для {feature}...")
            
            # Создаем модель для текущего параметра
            input_shape = (X_dict[feature].shape[1], X_dict[feature].shape[2])
            num_stations = len(self.station_encoder.classes_)
            model = self.create_feature_model(input_shape, num_stations, feature)
            self.models[feature] = model
            
            callbacks = [
                CustomProgressCallback(feature),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_mae',
                    patience=10,  # Увеличиваем терпение
                    restore_best_weights=True,
                    min_delta=1e-5
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/best_model_{feature}.keras',
                    monitor='val_mae',
                    save_best_only=True,
                    mode='min'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_mae',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                tf.keras.callbacks.CSVLogger(
                    f'logs/training_log_{feature}.csv',
                    separator=',',
                    append=False
                )
            ]
            
            # Обучаем модель
            history = model.fit(
                {
                    'time_series_input': X_dict[feature],
                    'station_input': station_ids,
                    'time_features_input': time_features_data
                },
                y_dict[feature],
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            histories[feature] = history
        
        return histories
    
    def predict(self, X_dict, station_codes, time_features_data):
        """
        Предсказание значений для каждого параметра
        """
        print(f"Коды станций для предсказания: {sorted(np.unique(station_codes))}")
        print(f"Известные коды станций: {sorted(self.station_encoder.classes_)}")
        
        # Проверяем наличие неизвестных станций
        unknown_stations = set(station_codes) - set(self.station_encoder.classes_)
        if unknown_stations:
            raise ValueError(f"Обнаружены неизвестные станции: {unknown_stations}")
        
        # Преобразуем коды станций
        station_ids = self.station_encoder.transform(station_codes)
        
        # Получаем предсказания для каждого параметра
        predictions = {}
        for feature in self.features:
            # Получаем предсказания
            pred = self.models[feature].predict({
                'time_series_input': X_dict[feature],
                'station_input': station_ids,
                'time_features_input': time_features_data
            })
            
            # Преобразование предсказаний обратно в исходный масштаб
            # Используем максимальное значение из диапазона нормы для обратного преобразования
            normal_ranges = {
                'SO2': 0.05,
                'NO2': 0.06,
                'CO': 9.0,
                'O3': 0.09,
                'PM10': 80.0,
                'PM2.5': 35.0
            }
            predictions[feature] = pred.flatten() * normal_ranges[feature]
        
        # Создаем DataFrame с результатами
        result_df = pd.DataFrame(predictions)
        result_df['Station code'] = station_codes
        return result_df 