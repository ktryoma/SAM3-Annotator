from __future__ import annotations
import os
import copy
import shutil
import random
from pathlib import Path
from typing import List, Optional, Dict

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QStatusBar, QSizePolicy, QSpinBox, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap, QKeySequence, QShortcut

from .canvas import AnnotationCanvas
from .annotation import BoundingBox
from .annotation_io import (
    save_cache, load_cache, bbs_to_cache_entries,
    cache_entry_to_bbs, save_yolo, save_yolo_from_entries, load_yolo,
)
from .inference_worker import InferenceWorker, list_images
from .label_dialog import LabelDialog
from .export_dialog import DatasetExportDialog
from .panels.left_panel import LeftPanel
from .panels.right_panel import RightPanel

MODEL_PATH = str(Path(__file__).parent.parent / "model" / "sam3.pt")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Annotation Tool")
        self.resize(1340, 820)

        # State
        self._folder: Optional[str] = None
        self._images: List[str] = []
        self._filtered_images: List[str] = []
        self._current_idx: int = -1
        self._cache: Dict[str, list] = {}
        # Snapshot of cache at last explicit save (per image) for Cancel feature
        self._saved_snapshot: Dict[str, list] = {}
        self._label_list: List[str] = []
        self._worker: Optional[InferenceWorker] = None
        self._lang: str = "ja"

        self._build_ui()
        self._connect_signals()
        self._apply_shortcuts()

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left panel
        self.left_panel = LeftPanel()
        root.addWidget(self.left_panel)

        # Center (canvas + nav bar)
        center_w = QWidget()
        center_layout = QVBoxLayout(center_w)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.canvas = AnnotationCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.canvas)

        # Navigation bar
        nav = QWidget()
        nav.setObjectName("navBar")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(10, 6, 10, 6)
        nav_layout.setSpacing(6)

        self.btn_prev = QPushButton("◀  Prev")
        self.btn_prev.setObjectName("navBtn")
        self.btn_prev.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)

        self.lbl_nav = QLabel("— / —")
        self.lbl_nav.setObjectName("navLabel")
        self.lbl_nav.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.lbl_nav, 1)

        # Jump to index SpinBox
        lbl_jump = QLabel("ジャンプ:")
        lbl_jump.setObjectName("jumpLabel")
        nav_layout.addWidget(lbl_jump)

        self.spin_nav = QSpinBox()
        self.spin_nav.setObjectName("navSpinBox")
        self.spin_nav.setMinimum(1)
        self.spin_nav.setMaximum(1)
        self.spin_nav.setValue(1)
        self.spin_nav.setEnabled(False)
        self.spin_nav.setFixedWidth(70)
        nav_layout.addWidget(self.spin_nav)

        self.btn_next = QPushButton("Next  ▶")
        self.btn_next.setObjectName("navBtn")
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)

        nav_layout.addSpacing(8)

        # Filter CheckBox
        self.chk_filter = QCheckBox("検出あり画像のみ")
        self.chk_filter.setObjectName("filterCheckBox")
        self.chk_filter.setChecked(False)
        self.chk_filter.setEnabled(False)
        self.chk_filter.setToolTip("SAM3推論やYOLO保存によりアノテーションが1つ以上存在する画像のみを絞り込んで表示します")
        nav_layout.addWidget(self.chk_filter)

        nav_layout.addSpacing(8)

        # ─ Action buttons ────────────────────────────────────────
        self.btn_cancel = QPushButton("↩  編集キャンセル")
        self.btn_cancel.setObjectName("cancelBtn")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("現在の画像の編集を破棄して最後に保存した状態に戻す")
        nav_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("💾  Save")
        self.btn_save.setObjectName("saveBtn")
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip("現在の画像をYOLO形式で保存 (Ctrl+S)")
        nav_layout.addWidget(self.btn_save)

        self.btn_save_all = QPushButton("💾  Save All")
        self.btn_save_all.setObjectName("saveAllBtn")
        self.btn_save_all.setEnabled(False)
        self.btn_save_all.setToolTip("全画像のアノテーションを一括YOLO保存")
        nav_layout.addWidget(self.btn_save_all)

        center_layout.addWidget(nav)
        root.addWidget(center_w, 1)

        # Right panel
        self.right_panel = RightPanel()
        root.addWidget(self.right_panel)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("フォルダを選択して開始してください")

    def _connect_signals(self):
        # Left panel
        self.left_panel.sig_open_folder.connect(self._open_folder)
        self.left_panel.sig_run_inference.connect(self._run_inference)
        self.left_panel.sig_labels_changed.connect(self._on_labels_changed)
        self.left_panel.sig_export_dataset.connect(self._export_dataset)

        # Canvas
        self.canvas.sig_bb_selected.connect(self._on_bb_selected)
        self.canvas.sig_bb_created.connect(self._on_bb_created)
        self.canvas.sig_bb_deleted.connect(self._on_bb_deleted)
        self.canvas.sig_bb_modified.connect(self._on_bb_modified)
        self.canvas.sig_label_requested.connect(self._on_label_requested)

        # Right panel
        self.right_panel.sig_item_clicked.connect(self.canvas.highlight_by_id)
        self.right_panel.sig_lang_changed.connect(self._change_language)

        # Nav + action buttons
        self.btn_prev.clicked.connect(self._go_prev)
        self.btn_next.clicked.connect(self._go_next)
        self.spin_nav.valueChanged.connect(self._on_spin_nav_changed)
        self.chk_filter.toggled.connect(self._on_filter_toggled)
        self.btn_save.clicked.connect(self._save_current)
        self.btn_save_all.clicked.connect(self._save_all)
        self.btn_cancel.clicked.connect(self._cancel_edits)

    def _apply_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_current)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self).activated.connect(self._save_all)
        QShortcut(QKeySequence("Left"), self).activated.connect(self._go_prev)
        QShortcut(QKeySequence("Right"), self).activated.connect(self._go_next)

    # ── folder & inference ────────────────────────────────────────

    @pyqtSlot()
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "画像フォルダを選択")
        if not folder:
            return
        self._folder = folder
        self._images = list_images(folder)
        self._cache = {}
        self._saved_snapshot = {}

        self.left_panel.set_folder_name(Path(folder).name)
        total = len(self._images)
        self.status.showMessage(f"{total} 枚の画像を検出")

        self.chk_filter.blockSignals(True)
        self.chk_filter.setChecked(False)
        self.chk_filter.setEnabled(total > 0)
        self.chk_filter.blockSignals(False)

        cached = load_cache(folder)
        if cached:
            self._cache = cached
            self._saved_snapshot = copy.deepcopy(cached)
            self.status.showMessage(
                f"キャッシュを読み込みました ({len(self._images)} 枚)")

        self._update_filtered_images()
        if self._filtered_images:
            self._load_image(0)
        else:
            self._load_image(0) if self._images else None

    @pyqtSlot(list)
    def _run_inference(self, label_entries: List[dict]):
        if not self._folder:
            return
        self._label_list = [e["label"] for e in label_entries]
        conf = self.left_panel.get_conf()
        self._worker = InferenceWorker(
            folder=self._folder,
            label_entries=label_entries,
            model_path=MODEL_PATH,
            conf=conf,
        )
        self._worker.progress.connect(self._on_inf_progress)
        self._worker.finished.connect(self._on_inf_finished)
        self._worker.error.connect(self._on_inf_error)
        self._worker.start()

    @pyqtSlot(int, int, str)
    def _on_inf_progress(self, current: int, total: int, fname: str):
        self.left_panel.set_progress(current, total, fname)
        self.status.showMessage(f"推論中… [{current}/{total}] {fname}")

    @pyqtSlot(dict)
    def _on_inf_finished(self, cache: dict):
        self._cache = cache
        self._saved_snapshot = copy.deepcopy(cache)
        save_cache(self._folder, cache)
        self.left_panel.set_inference_done()

        # Count detected labels
        label_counts = {}
        for entries in cache.values():
            for e in entries:
                label = e.get("label")
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1

        self.left_panel.set_inference_result(label_counts)
        self.status.showMessage("推論完了 ✓  キャッシュを保存しました")
        
        self._update_filtered_images()
        if self._filtered_images:
            self._load_image(0)
        else:
            self._load_image(0) if self._images else None

    @pyqtSlot(str)
    def _on_inf_error(self, msg: str):
        self.left_panel.set_inference_done()
        QMessageBox.critical(self, "推論エラー", msg)

    @pyqtSlot(list)
    def _on_labels_changed(self, labels: List[str]):
        self._label_list = labels

    # ── image navigation ──────────────────────────────────────────

    def _load_image(self, idx: int):
        """Load image at idx. Auto-syncs current canvas to in-memory cache first."""
        if not (0 <= idx < len(self._filtered_images)):
            return

        # Auto-sync current canvas to in-memory cache (no file write, no dialog)
        self._sync_canvas_to_cache()

        self._current_idx = idx
        img_path = self._filtered_images[idx]
        fname = Path(img_path).name

        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self.status.showMessage(f"画像の読み込みに失敗: {fname}")
            return

        # Load from cache or existing YOLO txt
        bbs: List[BoundingBox] = []
        if fname in self._cache:
            bbs = cache_entry_to_bbs(self._cache[fname],
                                     pixmap.width(), pixmap.height())
        else:
            txt_path = str(Path(img_path).parent / (Path(img_path).stem + ".txt"))
            if os.path.exists(txt_path):
                bbs = load_yolo(txt_path, self._label_list)
                # Put into cache so subsequent navigations stay consistent
                self._cache[fname] = bbs_to_cache_entries(bbs)

        self.canvas.load_image(pixmap, bbs)
        self.right_panel.refresh(bbs)

        total = len(self._filtered_images)
        self.lbl_nav.setText(f"{idx + 1} / {total}")
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < total - 1)

        self.spin_nav.blockSignals(True)
        self.spin_nav.setValue(idx + 1)
        self.spin_nav.blockSignals(False)

        self.btn_save.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_save_all.setEnabled(True)

        self.setWindowTitle(f"SAM3 Annotation — {fname}")
        self.status.showMessage(f"{fname}  ({len(bbs)} 件 of BB)")

    def _sync_canvas_to_cache(self):
        """Push current canvas BBs into in-memory cache (no disk write)."""
        if self._current_idx < 0 or self._current_idx >= len(self._filtered_images):
            return
        img_path = self._filtered_images[self._current_idx]
        fname = Path(img_path).name
        bbs = self.canvas.get_annotations()
        self._cache[fname] = bbs_to_cache_entries(bbs)

    def _go_prev(self):
        self._load_image(self._current_idx - 1)

    def _go_next(self):
        self._load_image(self._current_idx + 1)

    @pyqtSlot(int)
    def _on_spin_nav_changed(self, value: int):
        self._load_image(value - 1)

    @pyqtSlot(bool)
    def _on_filter_toggled(self, checked: bool):
        if not self._images:
            return

        # Keep track of the currently loaded image name to restore position
        current_fname = None
        if 0 <= self._current_idx < len(self._filtered_images):
            current_fname = Path(self._filtered_images[self._current_idx]).name

        self._update_filtered_images()

        new_idx = 0
        if current_fname:
            for idx, img_path in enumerate(self._filtered_images):
                if Path(img_path).name == current_fname:
                    new_idx = idx
                    break

        if self._filtered_images:
            self._load_image(new_idx)
        else:
            # No matching images with detections
            self.canvas.clear()
            self.right_panel.refresh([])
            self.lbl_nav.setText("0 / 0")
            self.spin_nav.blockSignals(True)
            self.spin_nav.setEnabled(False)
            self.spin_nav.setMaximum(1)
            self.spin_nav.setValue(1)
            self.spin_nav.blockSignals(False)
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self._current_idx = -1
            self.status.showMessage("検出されたオブジェクトを持つ画像はありません")

    def _update_filtered_images(self):
        if not self.chk_filter.isChecked():
            self._filtered_images = list(self._images)
        else:
            self._filtered_images = []
            for img in self._images:
                fname = Path(img).name
                has_bb = False
                if fname in self._cache and len(self._cache[fname]) > 0:
                    has_bb = True
                else:
                    # Fallback to YOLO txt existance with content
                    txt_path = Path(img).parent / (Path(img).stem + ".txt")
                    if txt_path.exists() and txt_path.stat().st_size > 0:
                        has_bb = True
                if has_bb:
                    self._filtered_images.append(img)

        # Update spinbox range
        total = len(self._filtered_images)
        self.spin_nav.blockSignals(True)
        self.spin_nav.setMaximum(total if total > 0 else 1)
        self.spin_nav.setEnabled(total > 0)
        self.spin_nav.blockSignals(False)

    # ── canvas slots ──────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_bb_selected(self, bb: Optional[BoundingBox]):
        if bb:
            self.right_panel.select_by_id(bb.bb_id)
            self.status.showMessage(
                f"選択: {bb.label or '(no label)'}  conf={bb.confidence:.2f}"
            )
        else:
            self.right_panel.select_by_id(None)

    @pyqtSlot(object)
    def _on_bb_created(self, bb: BoundingBox):
        self._ask_label(bb)
        self._refresh_right()

    @pyqtSlot(object)
    def _on_bb_deleted(self, bb: BoundingBox):
        self._refresh_right()
        self.status.showMessage(f"削除: {bb.label or '(no label)'}")

    @pyqtSlot(object)
    def _on_bb_modified(self, bb: BoundingBox):
        pass  # no dirty flag needed

    @pyqtSlot(object)
    def _on_label_requested(self, item):
        bb = item.bb
        self._ask_label(bb)
        self.canvas.update_item(bb.bb_id)
        self._refresh_right()

    def _ask_label(self, bb: BoundingBox):
        dlg = LabelDialog(self._label_list, current=bb.label, parent=self)
        label = dlg.get_label()
        if label is not None:
            bb.label = label
            if label not in self._label_list:
                self._label_list.append(label)
                self.left_panel.add_label_if_new(label, description="")
            self.canvas.update_item(bb.bb_id)
            self._refresh_right()

    def _refresh_right(self):
        bbs = self.canvas.get_annotations()
        self.right_panel.refresh(bbs)

    # ── save / cancel ─────────────────────────────────────────────

    def _save_current(self):
        """Save current image to YOLO txt + update cache + save snapshot."""
        if self._current_idx < 0 or self._current_idx >= len(self._filtered_images):
            return
        img_path = self._filtered_images[self._current_idx]
        fname = Path(img_path).name
        bbs = self.canvas.get_annotations()

        # Sync to in-memory cache
        entries = bbs_to_cache_entries(bbs)
        self._cache[fname] = entries

        # Save snapshot for cancel
        self._saved_snapshot[fname] = copy.deepcopy(entries)

        # Persist cache and YOLO txt
        save_cache(self._folder, self._cache)
        out = save_yolo(img_path, bbs, self._label_list)
        self.status.showMessage(f"保存完了: {out}")

        # Update filtered images list in case this image got its first annotation
        self._update_filtered_images()

    def _save_all(self):
        """
        Save YOLO txt for every image that has cache entries.
        First syncs current canvas to cache, then writes all files.
        """
        if not self._folder or not self._images:
            return

        # Sync current canvas first
        self._sync_canvas_to_cache()
        save_cache(self._folder, self._cache)

        # Update snapshot for all
        self._saved_snapshot = copy.deepcopy(self._cache)

        count = 0
        for img_path in self._images:
            fname = Path(img_path).name
            if fname in self._cache and self._cache[fname]:
                save_yolo_from_entries(img_path, self._cache[fname], self._label_list)
                count += 1

        self.status.showMessage(
            f"一括保存完了 ✓  {count} 枚のYOLOファイルを保存しました  (Ctrl+Shift+S)")

        # Update filtered images list
        self._update_filtered_images()

    def _cancel_edits(self):
        """Discard canvas changes; restore from last saved snapshot."""
        if self._current_idx < 0 or self._current_idx >= len(self._filtered_images):
            return
        img_path = self._filtered_images[self._current_idx]
        fname = Path(img_path).name

        # Restore cache from snapshot
        if fname in self._saved_snapshot:
            self._cache[fname] = copy.deepcopy(self._saved_snapshot[fname])
        else:
            self._cache[fname] = []

        # Reload display
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            return
        bbs = cache_entry_to_bbs(self._cache[fname], pixmap.width(), pixmap.height())
        self.canvas.load_image(pixmap, bbs)
        self.right_panel.refresh(bbs)
        self.status.showMessage(
            f"編集をキャンセルしました  ({fname}  {len(bbs)} 件のBB)")

    @pyqtSlot()
    def _export_dataset(self):
        if not self._folder or not self._images:
            QMessageBox.warning(self, "エラー", "画像フォルダが読み込まれていません。")
            return

        dlg = DatasetExportDialog(self)
        res = dlg.get_config()
        if not res:
            return

        export_path, train_ratio = res
        if not export_path:
            QMessageBox.warning(self, "エラー", "出力先フォルダを指定してください。")
            return

        # 1. Gather images that actually have annotations
        valid_images = []
        for img_path in self._images:
            fname = Path(img_path).name
            has_bb = False
            # Check memory cache
            if fname in self._cache and len(self._cache[fname]) > 0:
                has_bb = True
            else:
                # Check existing disk YOLO text file
                txt_path = Path(img_path).parent / (Path(img_path).stem + ".txt")
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    has_bb = True
            
            if has_bb:
                valid_images.append(img_path)

        if not valid_images:
            QMessageBox.warning(
                self, "データセット出力", 
                "アノテーション(境界ボックス)が存在する画像が見つかりませんでした。\n"
                "まずは推論を実行するか、手動でアノテーションを追加して保存してください。"
            )
            return

        # 2. Shuffle and split
        img_list = list(valid_images)
        random.seed(42)  # For reproducibility
        random.shuffle(img_list)

        split_idx = int(len(img_list) * train_ratio)
        # Ensure at least 1 image is in val if there are multiple images
        if split_idx == len(img_list) and len(img_list) > 1:
            split_idx = len(img_list) - 1
        # Ensure at least 1 image is in train
        if split_idx == 0 and len(img_list) > 0:
            split_idx = 1

        train_imgs = img_list[:split_idx]
        val_imgs = img_list[split_idx:]

        # 3. Create folders
        dirs = {
            "train_img": Path(export_path) / "train" / "images",
            "train_lbl": Path(export_path) / "train" / "labels",
            "val_img": Path(export_path) / "val" / "images",
            "val_lbl": Path(export_path) / "val" / "labels",
        }

        try:
            for d in dirs.values():
                d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"フォルダの作成に失敗しました:\n{e}")
            return

        # 4. Copy helper function
        def copy_data(imgs, img_dir, lbl_dir):
            copied_count = 0
            for img_path in imgs:
                p_img = Path(img_path)
                fname = p_img.name
                stem = p_img.stem

                # Destination paths
                dest_img_path = img_dir / fname
                dest_lbl_path = lbl_dir / f"{stem}.txt"

                # Copy image
                try:
                    shutil.copy2(img_path, dest_img_path)
                except Exception as e:
                    print(f"Failed to copy image {img_path}: {e}")
                    continue

                # Copy/Generate label txt
                # If it's in memory cache, we generate it directly from memory to ensure any unsaved (but in-memory) edits are included
                if fname in self._cache:
                    save_yolo_from_entries(str(dest_img_path), self._cache[fname], self._label_list, out_dir=str(lbl_dir))
                else:
                    # Otherwise copy existing file
                    src_txt = p_img.parent / f"{stem}.txt"
                    if src_txt.exists():
                        shutil.copy2(src_txt, dest_lbl_path)
                copied_count += 1
            return copied_count

        # Execute copying
        train_count = copy_data(train_imgs, dirs["train_img"], dirs["train_lbl"])
        val_count = copy_data(val_imgs, dirs["val_img"], dirs["val_lbl"])

        # 5. Generate dataset.yaml
        yaml_path = Path(export_path) / "dataset.yaml"
        try:
            yaml_content = []
            yaml_content.append(f"path: {Path(export_path).absolute().as_posix()}")
            yaml_content.append("train: train/images")
            yaml_content.append("val: val/images")
            yaml_content.append("")
            yaml_content.append("names:")
            for idx, label in enumerate(self._label_list):
                yaml_content.append(f"  {idx}: {label}")

            yaml_path.write_text("\n".join(yaml_content), encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"dataset.yaml の出力に失敗しました:\n{e}")

        # Show success message
        msg = (
            f"YOLO学習データの出力が完了しました！ ✓\n\n"
            f"■ 出力先: {export_path}\n"
            f"■ 分割数: Train: {train_count} 枚 / Val: {val_count} 枚 (総計 {train_count + val_count} 枚)\n\n"
            f"このフォルダ内の `dataset.yaml` を使用して、すぐにYOLOの学習を開始できます。"
        )
        QMessageBox.information(self, "出力完了", msg)

    @pyqtSlot(str)
    def _change_language(self, lang: str):
        self._lang = lang
        # Notify left panel
        self.left_panel.set_language(lang)

        # Update main window navigation bar UI
        if lang == "en":
            self.btn_prev.setText("◀  Prev")
            self.btn_next.setText("Next  ▶")
            self.chk_filter.setText("Detected Only")
            self.btn_cancel.setText("↩  Cancel Edits")
            self.btn_save.setText("💾  Save")
            self.btn_save_all.setText("💾  Save All")
            
            # Update tooltips
            self.chk_filter.setToolTip("Filter to show only images that have at least one annotation")
            self.btn_cancel.setToolTip("Discard current edits and revert to last saved state")
            self.btn_save.setToolTip("Save annotations for the current image in YOLO format (Ctrl+S)")
            self.btn_save_all.setToolTip("Save annotations for all cached images in YOLO format")
        else:
            self.btn_prev.setText("◀  Prev")
            self.btn_next.setText("Next  ▶")
            self.chk_filter.setText("検出あり画像のみ")
            self.btn_cancel.setText("↩  編集キャンセル")
            self.btn_save.setText("💾  Save")
            self.btn_save_all.setText("💾  Save All")

            # Update tooltips
            self.chk_filter.setToolTip("SAM3推論やYOLO保存によりアノテーションが1つ以上存在する画像のみを絞り込んで表示します")
            self.btn_cancel.setToolTip("現在の画像の編集を破棄して最後に保存した状態に戻す")
            self.btn_save.setToolTip("現在の画像をYOLO形式で保存 (Ctrl+S)")
            self.btn_save_all.setToolTip("全画像のアノテーションを一括YOLO保存")

        # Force refresh right panel label translation
        self.right_panel.refresh(self.canvas.get_annotations())

    # ── close ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        event.accept()
