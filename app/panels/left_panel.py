from __future__ import annotations
from typing import List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QLineEdit,
    QProgressBar, QGroupBox, QHeaderView, QAbstractItemView,
    QDoubleSpinBox, QSlider,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class LeftPanel(QWidget):
    """
    Left panel: folder selection, class label + description management,
    and SAM3 inference trigger.

    Each entry has:
      - label (str): YOLO class name (e.g. "chair")
      - description (str): SAM3 text prompt (e.g. "a wooden chair with four legs")
    """

    sig_open_folder = pyqtSignal()
    sig_run_inference = pyqtSignal(list)   # emits list of {"label":…, "description":…}
    sig_labels_changed = pyqtSignal(list)  # emits list of label strings
    sig_export_dataset = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Folder ──────────────────────────────────────────────
        self.grp_folder = QGroupBox("フォルダ")
        self.grp_folder.setObjectName("groupBox")
        fl = QVBoxLayout(self.grp_folder)
        fl.setSpacing(6)

        self.lbl_folder = QLabel("未選択")
        self.lbl_folder.setWordWrap(True)
        self.lbl_folder.setObjectName("folderLabel")
        fl.addWidget(self.lbl_folder)

        self.btn_open = QPushButton("📂  フォルダを開く")
        self.btn_open.setObjectName("primaryBtn")
        self.btn_open.clicked.connect(self.sig_open_folder)
        fl.addWidget(self.btn_open)

        layout.addWidget(self.grp_folder)

        # ── Class Labels with descriptions ──────────────────────
        self.grp_labels = QGroupBox("クラスラベル / 説明")
        self.grp_labels.setObjectName("groupBox")
        ll = QVBoxLayout(self.grp_labels)
        ll.setSpacing(6)

        # Table: label | description
        self.table = QTableWidget(0, 2)
        self.table.setObjectName("labelTable")
        self.table.setHorizontalHeaderLabels(["ラベル", "説明 (プロンプト)"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.SelectedClicked)
        self.table.setMinimumHeight(120)
        self.table.setMaximumHeight(200)
        self.table.itemChanged.connect(self._on_table_changed)
        ll.addWidget(self.table)

        # Input row: label name
        self.lbl_name_lbl = QLabel("ラベル名:")
        self.lbl_name_lbl.setObjectName("inputLabel")
        ll.addWidget(self.lbl_name_lbl)

        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText("例: chair")
        self.edit_label.setObjectName("lineEdit")
        self.edit_label.returnPressed.connect(self._focus_desc)
        ll.addWidget(self.edit_label)

        # Input row: description
        self.lbl_desc_lbl = QLabel("説明 (プロンプト):")
        self.lbl_desc_lbl.setObjectName("inputLabel")
        ll.addWidget(self.lbl_desc_lbl)

        self.edit_desc = QLineEdit()
        self.edit_desc.setPlaceholderText("例: a wooden chair with four legs")
        self.edit_desc.setObjectName("lineEdit")
        self.edit_desc.returnPressed.connect(self._add_entry)
        ll.addWidget(self.edit_desc)

        # Add / Remove buttons
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("+ 追加")
        self.btn_add.setObjectName("iconBtn")
        self.btn_add.clicked.connect(self._add_entry)
        btn_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton("− 削除")
        self.btn_remove.setObjectName("iconBtn")
        self.btn_remove.clicked.connect(self._remove_entry)
        btn_row.addWidget(self.btn_remove)
        ll.addLayout(btn_row)

        layout.addWidget(self.grp_labels)

        # ── Inference ───────────────────────────────────────────
        self.grp_inf = QGroupBox("SAM3 推論")
        self.grp_inf.setObjectName("groupBox")
        il = QVBoxLayout(self.grp_inf)
        il.setSpacing(6)

        self.btn_infer = QPushButton("▶  推論を実行")
        self.btn_infer.setObjectName("inferBtn")
        self.btn_infer.setEnabled(False)
        self.btn_infer.clicked.connect(self._on_infer_clicked)
        il.addWidget(self.btn_infer)

        # Confidence threshold
        conf_row = QHBoxLayout()
        self.conf_lbl = QLabel("検出 conf:")
        self.conf_lbl.setObjectName("inputLabel")
        conf_row.addWidget(self.conf_lbl)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setObjectName("confSpin")
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setToolTip(
            "SAM3の検出信頼度閾値 (低い→多く検出, 高い→精度重視)")
        conf_row.addWidget(self.conf_spin)
        il.addLayout(conf_row)

        # Conf visual slider (mirrors spinbox)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setObjectName("confSlider")
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_spin.setValue(v / 100))
        self.conf_spin.valueChanged.connect(
            lambda v: self.conf_slider.setValue(int(v * 100)))
        il.addWidget(self.conf_slider)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        il.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("")
        self.lbl_progress.setObjectName("progressLabel")
        self.lbl_progress.setVisible(False)
        self.lbl_progress.setWordWrap(True)
        il.addWidget(self.lbl_progress)

        self.lbl_result = QLabel("")
        self.lbl_result.setObjectName("resultLabel")
        self.lbl_result.setVisible(False)
        self.lbl_result.setWordWrap(True)
        il.addWidget(self.lbl_result)

        self.btn_export = QPushButton("📦 学習データを出力")
        self.btn_export.setObjectName("exportBtn")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.sig_export_dataset.emit)
        il.addWidget(self.btn_export)

        layout.addWidget(self.grp_inf)
        layout.addStretch()

    # ── public ────────────────────────────────────────────────────

    def get_conf(self) -> float:
        """Return current confidence threshold."""
        return self.conf_spin.value()

    def set_language(self, lang: str):
        if lang == "en":
            self.grp_folder.setTitle("Folder")
            self.btn_open.setText("📂  Open Folder")
            self.grp_labels.setTitle("Class Labels / Prompt")
            self.lbl_name_lbl.setText("Label name:")
            self.lbl_desc_lbl.setText("Description (prompt):")
            self.btn_add.setText("+ Add")
            self.btn_remove.setText("− Remove")
            self.grp_inf.setTitle("SAM3 Inference")
            if self.btn_infer.text() == "▶  推論を実行":
                self.btn_infer.setText("▶  Run Inference")
            elif self.btn_infer.text() == "推論中…":
                self.btn_infer.setText("Inference...")
            self.conf_lbl.setText("Detect conf:")
            self.btn_export.setText("📦 Export Dataset")
            self.table.setHorizontalHeaderLabels(["Label", "Description (Prompt)"])
        else:
            self.grp_folder.setTitle("フォルダ")
            self.btn_open.setText("📂  フォルダを開く")
            self.grp_labels.setTitle("クラスラベル / 説明")
            self.lbl_name_lbl.setText("ラベル名:")
            self.lbl_desc_lbl.setText("説明 (プロンプト):")
            self.btn_add.setText("+ 追加")
            self.btn_remove.setText("− 削除")
            self.grp_inf.setTitle("SAM3 推論")
            if self.btn_infer.text() == "▶  Run Inference":
                self.btn_infer.setText("▶  推論を実行")
            elif self.btn_infer.text() == "Inference...":
                self.btn_infer.setText("推論中…")
            self.conf_lbl.setText("検出 conf:")
            self.btn_export.setText("📦 学習データを出力")
            self.table.setHorizontalHeaderLabels(["ラベル", "説明 (プロンプト)"])

    def set_folder_name(self, name: str):
        self.lbl_folder.setText(name)
        self.btn_infer.setEnabled(True)
        self.btn_export.setEnabled(True)

    def get_entries(self) -> List[dict]:
        """Return list of {label, description} dicts."""
        entries = []
        for row in range(self.table.rowCount()):
            lbl_item = self.table.item(row, 0)
            desc_item = self.table.item(row, 1)
            label = lbl_item.text().strip() if lbl_item else ""
            desc = desc_item.text().strip() if desc_item else ""
            if label:
                entries.append({"label": label, "description": desc or label})
        return entries

    def get_labels(self) -> List[str]:
        """Return only label names (for YOLO output etc.)."""
        return [e["label"] for e in self.get_entries()]

    def add_label_if_new(self, label: str, description: str = ""):
        """Add a new label+description row if the label doesn't exist yet."""
        existing = self.get_labels()
        if label and label not in existing:
            self._insert_row(label, description or label)
            self.sig_labels_changed.emit(self.get_labels())

    def set_progress(self, current: int, total: int, fname: str):
        self.progress_bar.setVisible(True)
        self.lbl_progress.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_progress.setText(f"[{current}/{total}] {fname}")

    def set_inference_result(self, counts: dict):
        if not counts:
            self.lbl_result.setText("推論完了 ✓\n検出されたオブジェクトはありません")
        else:
            lines = ["推論完了 ✓ (検出数):"]
            for lbl, cnt in counts.items():
                lines.append(f"  • {lbl}: {cnt} 件")
            self.lbl_result.setText("\n".join(lines))
        self.lbl_result.setVisible(True)

    def set_inference_done(self):
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)
        self.btn_infer.setEnabled(True)
        self.btn_infer.setText("▶  推論を実行")

    def set_inference_running(self):
        self.lbl_result.setVisible(False)
        self.btn_infer.setEnabled(False)
        self.btn_infer.setText("推論中…")
        self.progress_bar.setVisible(True)
        self.lbl_progress.setVisible(True)

    # ── private ───────────────────────────────────────────────────

    def _focus_desc(self):
        self.edit_desc.setFocus()

    def _insert_row(self, label: str, description: str):
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(label))
        self.table.setItem(row, 1, QTableWidgetItem(description))
        self.table.blockSignals(False)

    def _add_entry(self):
        label = self.edit_label.text().strip()
        desc = self.edit_desc.text().strip()
        if not label:
            self.edit_label.setFocus()
            return
        if label in self.get_labels():
            # Update description if label already exists
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item and item.text() == label:
                    self.table.blockSignals(True)
                    self.table.setItem(row, 1, QTableWidgetItem(desc or label))
                    self.table.blockSignals(False)
                    break
        else:
            self._insert_row(label, desc or label)
        self.edit_label.clear()
        self.edit_desc.clear()
        self.edit_label.setFocus()
        self.sig_labels_changed.emit(self.get_labels())

    def _remove_entry(self):
        rows = set(i.row() for i in self.table.selectedItems())
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)
        self.sig_labels_changed.emit(self.get_labels())

    def _on_table_changed(self, item):
        self.sig_labels_changed.emit(self.get_labels())

    def _on_infer_clicked(self):
        entries = self.get_entries()
        if not entries:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告",
                                "クラスラベルを少なくとも1つ入力してください。")
            return
        # Validate: warn about entries with no description
        for e in entries:
            if not e["description"] or e["description"] == e["label"]:
                pass  # will use label as fallback prompt
        self.set_inference_running()
        self.sig_run_inference.emit(entries)
