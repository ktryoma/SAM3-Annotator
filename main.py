"""
SAM3 Annotation Tool
Entry point.
"""
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from app.main_window import MainWindow

STYLE_PATH = Path(__file__).parent / "assets" / "style.qss"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SAM3 Annotation Tool")
    app.setOrganizationName("sam3ano")

    if STYLE_PATH.exists():
        app.setStyleSheet(STYLE_PATH.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
