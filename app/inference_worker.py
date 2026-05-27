from __future__ import annotations
from pathlib import Path
from typing import List

from PyQt6.QtCore import QThread, pyqtSignal


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: str) -> List[str]:
    return sorted(
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )


class InferenceWorker(QThread):
    """
    Background thread: runs SAM3 on every image in folder.

    Each label entry has:
      - "label": str       → YOLO class name (assigned to detected boxes)
      - "description": str → SAM3 text prompt (used for detection)

    Results are stored in cache JSON (bboxes only, no masks).
    """
    progress = pyqtSignal(int, int, str)   # current, total, filename
    finished = pyqtSignal(dict)            # {filename: [entry, ...]}
    error = pyqtSignal(str)

    def __init__(self, folder: str, label_entries: List[dict],
                 model_path: str = "model/sam3.pt", conf: float = 0.25):
        super().__init__()
        self.folder = folder
        self.label_entries = label_entries   # [{"label": "chair", "description": "…"}, …]
        self.model_path = model_path
        self.conf = conf
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError as e:
            self.error.emit(f"ultralyticsのインポートに失敗しました: {e}")
            return

        images = list_images(self.folder)
        if not images:
            self.error.emit("画像が見つかりませんでした")
            return

        # Build prompt list (descriptions) and parallel label list
        descriptions = [e["description"] for e in self.label_entries]
        labels = [e["label"] for e in self.label_entries]

        overrides = dict(
            conf=self.conf,
            task="segment",
            mode="predict",
            model=self.model_path,
            imgsz=644,
            half=True,
            save=False,
            verbose=False,
        )

        try:
            predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            self.error.emit(f"SAM3モデルの読み込みに失敗しました: {e}")
            return

        cache: dict = {}
        total = len(images)

        for idx, img_path in enumerate(images):
            if self._stop:
                break
            fname = Path(img_path).name
            self.progress.emit(idx + 1, total, fname)

            try:
                predictor.set_image(img_path)
                # Use descriptions as text prompts
                results = predictor(text=descriptions)
            except Exception:
                cache[fname] = []
                continue

            entries = []
            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes
                img_w = result.orig_shape[1]
                img_h = result.orig_shape[0]

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().tolist()
                    conf = float(boxes.conf[i].cpu())
                    # cls_idx maps back to the description index → label name
                    cls_idx = int(boxes.cls[i].cpu()) if boxes.cls is not None else 0
                    label = labels[cls_idx] if cls_idx < len(labels) else ""

                    x1 = max(0.0, xyxy[0] / img_w)
                    y1 = max(0.0, xyxy[1] / img_h)
                    x2 = min(1.0, xyxy[2] / img_w)
                    y2 = min(1.0, xyxy[3] / img_h)

                    entries.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": label,
                        "conf": round(conf, 4),
                        "is_auto": True,
                    })

            cache[fname] = entries

        self.finished.emit(cache)
