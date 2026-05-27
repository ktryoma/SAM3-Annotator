from __future__ import annotations
from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel, QDialogButtonBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class LabelDialog(QDialog):
    """
    Dialog to assign a class label to a bounding box.
    Shows a combo box pre-populated with existing labels.
    User can also type a new label.
    """

    def __init__(self, labels: List[str], current: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("ラベルを選択 / 入力")
        self.setModal(True)
        self.setMinimumWidth(300)
        self._build_ui(labels, current)

    def _build_ui(self, labels: List[str], current: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        lbl = QLabel("クラスラベル:")
        lbl.setFont(QFont("Segoe UI", 10))
        layout.addWidget(lbl)

        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setObjectName("labelCombo")
        self.combo.addItems(labels)
        self.combo.setCurrentText(current)
        self.combo.lineEdit().selectAll()
        layout.addWidget(self.combo)

        if labels:
            hint = QLabel(f"既存のラベル: {', '.join(labels)}")
            hint.setObjectName("hintLabel")
            hint.setWordWrap(True)
            layout.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_label(self) -> Optional[str]:
        if self.exec() == QDialog.DialogCode.Accepted:
            return self.combo.currentText().strip() or None
        return None
