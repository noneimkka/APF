# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Qt5Agg')

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        
        # Создаем фигуру с нужным размером
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        
        # Настраиваем размеры
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setMinimumHeight(300)
        
        # Создаем панель инструментов
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setMaximumHeight(35)
        
        # Отключаем кнопку для создания нового окна
        actions = self.toolbar.actions()
        for action in actions:
            if action.text() == 'Customize':  # Отключаем кнопку настройки
                action.setVisible(False)
            elif action.text() == 'Save':  # Отключаем кнопку сохранения
                action.setVisible(False)
        
        # Добавляем виджеты в layout
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # Отключаем фокус для предотвращения перехвата клавиш
        self.canvas.setFocusPolicy(Qt.NoFocus)
        
        # Настраиваем обработку событий колеса мыши
        self.canvas.wheelEvent = self.wheelEvent
        
        # Отключаем контекстное меню
        self.canvas.setContextMenuPolicy(Qt.NoContextMenu)
    
    def get_figure(self):
        return self.figure
    
    def wheelEvent(self, event):
        # Передаем событие прокрутки родительскому виджету
        if self.parent():
            self.parent().wheelEvent(event)
        else:
            super().wheelEvent(event)
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # При изменении размера обновляем layout графика
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()