import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import AirPollutionApp

def main():
    app = QApplication(sys.argv)
    window = AirPollutionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 