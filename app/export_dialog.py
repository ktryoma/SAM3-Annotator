from __future__ import annotations
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QFileDialog, QDialogButtonBox,
)
from PyQt6.QtCore import Qt


class DatasetExportDialog(QDialog):
    """
    Dialog for configuring YOLO dataset export.
    Allows choosing:
      - Export destination directory
      - Train/Val split ratio (e.g. 80% / 20%)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO 学習データの出力設定")
        self.resize(450, 240)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Destination Path ──
        path_lbl = QLabel("出力先フォルダ:")
        path_lbl.setObjectName("inputLabel")
        layout.addWidget(path_lbl)

        path_row = QHBoxLayout()
        self.edit_path = QLineEdit()
        self.edit_path.setObjectName("lineEdit")
        self.edit_path.setPlaceholderText("データセットの保存先フォルダを選択してください")
        path_row.addWidget(self.edit_path)

        self.btn_browse = QPushButton("参照...")
        self.btn_browse.setObjectName("iconBtn")
        self.btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self.btn_browse)
        layout.addLayout(path_row)

        # ── Split Ratio ──
        ratio_lbl = QLabel("分割割合 (Train / Val):")
        ratio_lbl.setObjectName("inputLabel")
        layout.addWidget(ratio_lbl)

        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(10)

        # Train spinbox
        self.spin_train = QSpinBox()
        self.spin_train.setObjectName("navSpinBox")
        self.spin_train.setRange(50, 99)
        self.spin_train.setValue(80)
        self.spin_train.setSuffix(" % (Train)")
        self.spin_train.valueChanged.connect(self._update_val_ratio)
        ratio_row.addWidget(self.spin_train)

        # Val label (mirrors Train spinbox)
        self.lbl_val = QLabel("20 % (Val)")
        self.lbl_val.setObjectName("jumpLabel")
        self.lbl_val.setStyleSheet("font-weight: bold; color: #a0a0c0;")
        ratio_row.addWidget(self.lbl_val)
        ratio_row.addStretch()

        layout.addLayout(ratio_row)

        # ── Note ──
        note = QLabel("※ アノテーション(境界ボックス)が1つ以上存在する画像のみを抽出し、\n   指定した割合でランダムに分割して出力します。")
        note.setObjectName("hintLabel")
        note.setStyleSheet("color: #707090; font-size: 11px;")
        layout.addWidget(note)

        layout.addSpacing(10)

        # ── Dialog Buttons ──
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択")
        if folder:
            self.edit_path.setText(folder)

    def _update_val_ratio(self, train_val: int):
        val_val = 100 - train_val
        self.lbl_val.setText(f"{val_val} % (Val)")

    def get_config(self) -> tuple[str, float] | None:
        """Returns (export_path, train_ratio) or None if cancelled."""
        if self.exec() == QDialog.DialogCode.Accepted:
            path = self.edit_path.text().strip()
            train_ratio = self.spin_train.value() / 100.0
            return path, train_ratio
        return None
