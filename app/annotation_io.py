from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List

from .annotation import BoundingBox

CACHE_DIR = ".sam3_cache"
CACHE_FILE = "cache.json"


def get_cache_path(folder: str) -> Path:
    return Path(folder) / CACHE_DIR / CACHE_FILE


def save_cache(folder: str, data: Dict[str, List[dict]]) -> None:
    cache_path = get_cache_path(folder)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cache(folder: str) -> Dict[str, List[dict]] | None:
    cache_path = get_cache_path(folder)
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_entry_to_bbs(entries: List[dict], img_w: int, img_h: int) -> List[BoundingBox]:
    bbs = []
    for e in entries:
        x1, y1, x2, y2 = e["bbox"]
        bbs.append(BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            label=e.get("label", ""),
            confidence=e.get("conf", 0.0),
            is_auto=True,
        ))
    return bbs


def bbs_to_cache_entries(bbs: List[BoundingBox]) -> List[dict]:
    return [
        {
            "bbox": [bb.x1, bb.y1, bb.x2, bb.y2],
            "label": bb.label,
            "conf": bb.confidence,
            "is_auto": bb.is_auto,
        }
        for bb in bbs
    ]


def save_yolo(image_path: str, bbs: List[BoundingBox], label_list: List[str],
              out_dir: str | None = None) -> str:
    """Save YOLO format .txt file. Returns the output path."""
    img_p = Path(image_path)
    if out_dir:
        out_p = Path(out_dir) / (img_p.stem + ".txt")
    else:
        out_p = img_p.parent / (img_p.stem + ".txt")

    lines = []
    for bb in bbs:
        if not bb.label:
            continue
        try:
            cls_idx = label_list.index(bb.label)
        except ValueError:
            label_list.append(bb.label)
            cls_idx = len(label_list) - 1
        xc, yc, w, h = bb.normalized_xywh()
        lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(out_p)


def load_yolo(txt_path: str, label_list: List[str]) -> List[BoundingBox]:
    """Load YOLO .txt annotations back as BoundingBox list."""
    bbs = []
    p = Path(txt_path)
    if not p.exists():
        return bbs
    with open(p, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_idx, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
            label = label_list[cls_idx] if cls_idx < len(label_list) else str(cls_idx)
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label, is_auto=False))
    return bbs


def save_yolo_from_entries(image_path: str, entries: List[dict],
                           label_list: List[str],
                           out_dir: str | None = None) -> str:
    """
    Save YOLO format .txt from raw cache entries (no BoundingBox conversion needed).
    Returns the output path.
    """
    img_p = Path(image_path)
    if out_dir:
        out_p = Path(out_dir) / (img_p.stem + ".txt")
    else:
        out_p = img_p.parent / (img_p.stem + ".txt")

    lines = []
    for e in entries:
        label = e.get("label", "")
        if not label:
            continue
        try:
            cls_idx = label_list.index(label)
        except ValueError:
            label_list.append(label)
            cls_idx = len(label_list) - 1
        x1, y1, x2, y2 = e["bbox"]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(out_p)
