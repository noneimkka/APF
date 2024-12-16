# -*- coding: utf-8 -*-
from PySide6.QtWidgets import QWidget

class ScrollableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scroll_area = None
    
    def wheelEvent(self, event):
        if self.scroll_area:
            delta = event.angleDelta().y()
            vertical_bar = self.scroll_area.verticalScrollBar()
            vertical_bar.setValue(vertical_bar.value() - delta)
            event.accept()
        else:
            super().wheelEvent(event) 