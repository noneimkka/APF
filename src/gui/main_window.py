import pandas as pd
from datetime import datetime, timedelta
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QComboBox, QPushButton, QDateEdit,
                            QSpacerItem, QSizePolicy, QScrollArea, QMessageBox)
from PySide6.QtCore import Qt, QDate, QTime

from .widgets import ScrollableWidget
from .plot_widget import PlotWidget
from models.predictor import ModelPredictor

class AirPollutionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогноз загрязнения воздуха")
        self.setMinimumSize(600, 400)
        
        # Загружаем информацию о станциях
        self.stations_df = pd.read_csv('data/AirPollutionSeoul/Original Data/Measurement_station_info.csv')
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем верхнюю панель с элементами управ��ения
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Выбор станции
        station_label = QLabel("Станция:")
        self.station_combo = QComboBox()
        
        # Добавляем станции в выпадающий список
        station_items = [f"{row['Station name(district)']} (Код: {row['Station code']})" 
                        for _, row in self.stations_df.iterrows()]
        self.station_combo.addItems(station_items)
        
        # Выбор даты
        date_label = QLabel("Дата:")
        self.date_picker = QDateEdit()
        self.date_picker.setDisplayFormat("dd.MM.yyyy")
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())
        
        # Добавляем разделитель между датой и временем
        spacer = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
        
        # Выбор времени
        time_label = QLabel("Время:")
        self.time_combo = QComboBox()
        # Добавляем часы от 0 до 23
        self.time_combo.addItems([f"{hour:02d}:00" for hour in range(24)])
        # Устанавливаем текущий час
        current_hour = QTime.currentTime().hour()
        self.time_combo.setCurrentIndex(current_hour)
        
        # Кнопка прогноза
        self.predict_button = QPushButton("Рассчитать")
        self.predict_button.clicked.connect(self.make_prediction)
        
        # Добавляем элементы на панель управления
        control_layout.addWidget(station_label)
        control_layout.addWidget(self.station_combo)
        control_layout.addWidget(date_label)
        control_layout.addWidget(self.date_picker)
        control_layout.addSpacerItem(spacer)
        control_layout.addWidget(time_label)
        control_layout.addWidget(self.time_combo)
        control_layout.addWidget(self.predict_button)
        
        # Создаем область для отображения результатов с возможностью прокрутки
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Создаем виджет с возможностью прокрутки
        results_widget = ScrollableWidget()
        results_widget.scroll_area = scroll_area
        results_layout = QVBoxLayout(results_widget)
        results_layout.setSpacing(10)
        
        # Информация о станции
        self.station_info = QLabel()
        self.update_station_info()
        results_layout.addWidget(self.station_info)
        
        # Создаем виджет для прогноза
        prediction_widget = QWidget()
        prediction_layout = QVBoxLayout(prediction_widget)
        
        # Заголовок с временем
        self.prediction_time_label = QLabel()
        prediction_layout.addWidget(self.prediction_time_label)
        
        # Значения показателей
        self.prediction_values = QLabel("Нет данных")
        prediction_layout.addWidget(self.prediction_values)
        
        # Временной диапазон прогноза
        self.prediction_range_label = QLabel()
        prediction_layout.addWidget(self.prediction_range_label)
        
        results_layout.addWidget(prediction_widget)
        
        # Создаем контейнер для графиков с возможностью прокрутки
        self.graphs_container = ScrollableWidget()
        self.graphs_container.scroll_area = scroll_area
        self.graphs_layout = QVBoxLayout(self.graphs_container)
        
        # Добавляем ��тейнер графиков в основной layout
        results_layout.addWidget(self.graphs_container)
        
        # Устанавливаем widget в область прокрутки
        scroll_area.setWidget(results_widget)
        
        # Добавляем элементы в главный layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(scroll_area)
        
        # Подключаем обновление информации о станции при изменении выбора
        self.station_combo.currentIndexChanged.connect(self.update_station_info)
        
        # Обновляем заголовок показателей
        self.update_prediction_time_label()
        # Подключаем обновление заголовка при изменении даты/времени
        self.date_picker.dateChanged.connect(self.update_prediction_time_label)
        self.time_combo.currentIndexChanged.connect(self.update_prediction_time_label)
        
        # Инициализируем предиктор
        self.predictor = ModelPredictor()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # При изменении размера окна не пересоздаем графики
        if hasattr(self, 'graphs_container'):
            self.graphs_container.update()
    
    def update_station_info(self):
        current_index = self.station_combo.currentIndex()
        if current_index >= 0:
            station = self.stations_df.iloc[current_index]
            info_text = (f"Адрес: {station['Address']}\n"
                        f"Координаты: {station['Latitude']}, {station['Longitude']}")
            self.station_info.setText(info_text)
    
    def get_current_station_code(self):
        current_index = self.station_combo.currentIndex()
        if current_index >= 0:
            return self.stations_df.iloc[current_index]['Station code']
        return None
    
    def get_current_station_name(self):
        current_index = self.station_combo.currentIndex()
        if current_index >= 0:
            return self.stations_df.iloc[current_index]['Station name(district)']
        return None
    
    def get_selected_datetime(self):
        selected_date = self.date_picker.date().toPython()
        selected_hour = int(self.time_combo.currentText().split(':')[0])
        return datetime.combine(selected_date, datetime.min.time().replace(hour=selected_hour))
    
    def update_prediction_time_label(self):
        selected_datetime = self.get_selected_datetime()
        self.prediction_time_label.setText(
            f"Показатели на {selected_datetime.strftime('%d.%m.%Y %H:00')}"
        )
    
    def make_prediction(self):
        station_code = self.get_current_station_code()
        station_name = self.get_current_station_name()
        selected_datetime = self.get_selected_datetime()
        
        # Получаем временной диапазон для прогноза
        time_minus_2h = selected_datetime - timedelta(hours=2)
        time_plus_2h = selected_datetime + timedelta(hours=2)
        
        try:
            # Получаем предсказания от моделей
            predictions, time_points = self.predictor.predict_range(station_code, selected_datetime)
            
            # Проверяем, есть ли доступные предсказания
            available_params = self.predictor.get_available_parameters()
            if not available_params:
                QMessageBox.warning(self, "Ошибка", "Нет доступных моделей для предсказания")
                return
            
            # Обновляем диапазон прогноза
            self.prediction_range_label.setText(
                f"Прогноз для станции {station_name} на период\n"
                f"{time_minus_2h.strftime('%d.%m.%Y %H:00')} - {time_plus_2h.strftime('%d.%m.%Y %H:00')}"
            )
            
            # Формируем текст с предсказанными значениями для выбранного времени
            prediction_text = []
            for param in self.predictor.parameters:
                if param in available_params:
                    # Берем значение для выбранного времени (центральный индекс = 2)
                    value = predictions[param][2]
                    prediction_text.append(f"{param}: {value:.3f} мкг/м³")
                else:
                    prediction_text.append(f"{param}: нет данных")
            
            self.prediction_values.setText("\n".join(prediction_text))
            
            # Сохраняем данные для возможности перерисовки
            self.last_prediction_data = {
                'station_name': station_name,
                'datetime': selected_datetime,
                'parameters': available_params,
                'values': {param: predictions[param] for param in available_params}
            }
            
            # Обновляем графики
            self.update_graphs()
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при получении прогноза: {str(e)}")
    
    def update_graphs(self):
        # Очищаем контейнер графиков
        while self.graphs_layout.count():
            item = self.graphs_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        if not hasattr(self, 'last_prediction_data'):
            return
        
        # Словарь единиц измерения
        units = {
            'SO2': 'ppm',
            'NO2': 'ppm',
            'O3': 'ppm',
            'CO': 'ppm',
            'PM10': 'мкг/м³',
            'PM2.5': 'мкг/м³'
        }
        
        # Создаем новые графики
        for param in self.last_prediction_data['parameters']:
            values = self.last_prediction_data['values'][param]
            
            # Создаем виджет с графиком
            plot_widget = PlotWidget()
            fig = plot_widget.get_figure()
            ax = fig.add_subplot(111)
            
            # Создаем временные точки для графика
            time_points = [
                self.last_prediction_data['datetime'] + timedelta(hours=i)
                for i in range(-2, 3)
            ]
            
            # Форматируем метки времени с датой (в две строки)
            time_labels = [f"{t.strftime('%d.%m')}\n{t.strftime('%H:00')}" for t in time_points]
            
            # Строим график
            ax.plot(time_labels, values, 'o-', linewidth=2, markersize=8)
            ax.set_title(f"{param}", fontsize=12, pad=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Настраиваем оси
            ax.set_xlabel('Дата и время', fontsize=10)
            ax.set_ylabel(units[param], fontsize=10)
            
            # Настраиваем метки времени
            ax.tick_params(axis='x', labelsize=10, rotation=0)
            ax.tick_params(axis='y', labelsize=10)
            
            # Настраиваем внешний вид
            fig.tight_layout(pad=2.0)
            
            # Добавляем график в контейнер
            self.graphs_layout.addWidget(plot_widget)
        
        # Добавляем растягивающийся элемент в конец
        self.graphs_layout.addStretch()