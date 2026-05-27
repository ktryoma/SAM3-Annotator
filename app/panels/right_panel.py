from __future__ import annotations
from typing import List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QGroupBox, QPushButton,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor

from ..annotation import BoundingBox

COLOR_AUTO = "#50beff"
COLOR_MANUAL = "#64ff8c"


class RightPanel(QWidget):
    """Right panel: list of BBs in current image."""

    sig_item_clicked = pyqtSignal(str)  # bb_id
    sig_lang_changed = pyqtSignal(str)  # emits "ja" or "en"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self._bb_ids: List[str] = []
        self._lang = "ja"
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Language Toggle Button at the very top
        lang_layout = QHBoxLayout()
        lang_layout.addStretch()
        self.btn_lang = QPushButton("日本語")
        self.btn_lang.setCheckable(True)
        self.btn_lang.setObjectName("langBtn")
        self.btn_lang.setFixedWidth(80)
        self.btn_lang.setToolTip("言語を英語に切り替える / Switch to English")
        self.btn_lang.clicked.connect(self._on_lang_clicked)
        lang_layout.addWidget(self.btn_lang)
        layout.addLayout(lang_layout)

        self.grp = QGroupBox("アノテーション一覧")
        self.grp.setObjectName("groupBox")
        gl = QVBoxLayout(self.grp)

        self.lbl_count = QLabel("0 件")
        self.lbl_count.setObjectName("countLabel")
        gl.addWidget(self.lbl_count)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("bbList")
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        gl.addWidget(self.list_widget)

        layout.addWidget(self.grp)
        layout.addStretch()

    # ── public ────────────────────────────────────────────────────

    def refresh(self, bbs: List[BoundingBox]):
        self.list_widget.clear()
        self._bb_ids.clear()
        for bb in bbs:
            label = bb.label or "(no label)"
            src = "🤖" if bb.is_auto else "✏️"
            item = QListWidgetItem(f"{src} {label}")
            item.setData(Qt.ItemDataRole.UserRole, bb.bb_id)
            color = COLOR_AUTO if bb.is_auto else COLOR_MANUAL
            item.setForeground(QColor(color))
            self.list_widget.addItem(item)
            self._bb_ids.append(bb.bb_id)

        if self._lang == "en":
            self.lbl_count.setText(f"{len(bbs)} items")
        else:
            self.lbl_count.setText(f"{len(bbs)} 件")

    def select_by_id(self, bb_id: Optional[str]):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == bb_id:
                self.list_widget.setCurrentItem(item)
                return
        self.list_widget.clearSelection()

    def set_language(self, lang: str):
        self._lang = lang
        if lang == "en":
            self.btn_lang.setText("English")
            self.btn_lang.setChecked(True)
            self.btn_lang.setToolTip("Switch to Japanese")
            self.grp.setTitle("Annotations")
        else:
            self.btn_lang.setText("日本語")
            self.btn_lang.setChecked(False)
            self.btn_lang.setToolTip("英語に切り替える")
            self.grp.setTitle("アノテーション一覧")

    # ── private ───────────────────────────────────────────────────

    def _on_item_clicked(self, item: QListWidgetItem):
        bb_id = item.data(Qt.ItemDataRole.UserRole)
        self.sig_item_clicked.emit(bb_id)

    def _on_lang_clicked(self, checked: bool):
        lang = "en" if checked else "ja"
        self.set_language(lang)
        self.sig_lang_changed.emit(lang)
