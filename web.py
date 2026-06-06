import sys
import os
import copy
import json
import random
import shutil
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from app.inference_worker import list_images
from app.annotation_io import (
    save_cache, load_cache, bbs_to_cache_entries,
    cache_entry_to_bbs, save_yolo, save_yolo_from_entries, load_yolo
)
from app.annotation import BoundingBox

app = Flask(__name__)

# Global state for local single-user app
STATE = {
    "folder": None,
    "images": [],
    "filtered_images": [],
    "cache": {},
    "label_list": [],
    "is_inferencing": False,
    "inf_progress": 0,
    "inf_total": 0,
    "inf_msg": ""
}

MODEL_PATH = str(Path(__file__).parent / "model" / "sam3.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/open_folder", methods=["POST"])
def open_folder():
    data = request.json
    folder = data.get("folder")
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "無効なフォルダパスです"}), 400

    STATE["folder"] = folder
    STATE["images"] = list_images(folder)
    STATE["cache"] = load_cache(folder) or {}
    STATE["filtered_images"] = list(STATE["images"])
    
    return jsonify({
        "success": True,
        "total": len(STATE["images"]),
        "folder": folder
    })


@app.route("/api/images", methods=["GET"])
def get_images():
    filter_only = request.args.get("filter") == "true"
    
    if not filter_only:
        STATE["filtered_images"] = list(STATE["images"])
    else:
        filtered = []
        for img in STATE["images"]:
            fname = Path(img).name
            has_bb = False
            if fname in STATE["cache"] and len(STATE["cache"][fname]) > 0:
                has_bb = True
            else:
                txt_path = Path(img).parent / (Path(img).stem + ".txt")
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    has_bb = True
            if has_bb:
                filtered.append(img)
        STATE["filtered_images"] = filtered

    return jsonify({
        "total": len(STATE["filtered_images"]),
        "filenames": [Path(p).name for p in STATE["filtered_images"]]
    })


@app.route("/api/image/<int:idx>")
def serve_image(idx):
    if idx < 0 or idx >= len(STATE["filtered_images"]):
        return "Not found", 404
    img_path = STATE["filtered_images"][idx]
    return send_file(img_path)


@app.route("/api/annotations/<int:idx>", methods=["GET", "POST"])
def annotations(idx):
    if idx < 0 or idx >= len(STATE["filtered_images"]):
        return jsonify({"error": "Out of bounds"}), 400
        
    img_path = STATE["filtered_images"][idx]
    fname = Path(img_path).name

    if request.method == "GET":
        entries = []
        if fname in STATE["cache"]:
            entries = STATE["cache"][fname]
        else:
            txt_path = Path(img_path).parent / (Path(img_path).stem + ".txt")
            if txt_path.exists():
                bbs = load_yolo(str(txt_path), STATE["label_list"])
                entries = bbs_to_cache_entries(bbs)
                STATE["cache"][fname] = entries
        return jsonify({"annotations": entries})

    if request.method == "POST":
        data = request.json
        entries = data.get("annotations", [])
        STATE["cache"][fname] = entries
        
        # Save cache and YOLO
        if STATE["folder"]:
            save_cache(STATE["folder"], STATE["cache"])
            save_yolo_from_entries(img_path, entries, STATE["label_list"])
            
        return jsonify({"success": True})


@app.route("/api/save_all", methods=["POST"])
def save_all():
    if not STATE["folder"]:
        return jsonify({"error": "No folder opened"}), 400
        
    save_cache(STATE["folder"], STATE["cache"])
    count = 0
    for img_path in STATE["images"]:
        fname = Path(img_path).name
        if fname in STATE["cache"] and STATE["cache"][fname]:
            save_yolo_from_entries(img_path, STATE["cache"][fname], STATE["label_list"])
            count += 1
            
    return jsonify({"success": True, "count": count})


def run_inference_thread(folder, label_entries, conf):
    STATE["is_inferencing"] = True
    STATE["inf_msg"] = "モデル読み込み中..."
    
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as e:
        STATE["inf_msg"] = f"Error: {e}"
        STATE["is_inferencing"] = False
        return

    images = list_images(folder)
    if not images:
        STATE["inf_msg"] = "Error: 画像が見つかりません"
        STATE["is_inferencing"] = False
        return

    descriptions = [e["description"] for e in label_entries]
    labels = [e["label"] for e in label_entries]

    overrides = dict(
        conf=conf, task="segment", mode="predict", model=MODEL_PATH,
        imgsz=644, half=True, save=False, verbose=False,
    )

    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        STATE["inf_msg"] = f"Error: {e}"
        STATE["is_inferencing"] = False
        return

    total = len(images)
    STATE["inf_total"] = total
    
    for idx, img_path in enumerate(images):
        fname = Path(img_path).name
        STATE["inf_progress"] = idx + 1
        STATE["inf_msg"] = f"推論中: {fname}"

        try:
            predictor.set_image(img_path)
            results = predictor(text=descriptions)
        except Exception:
            if fname not in STATE["cache"]:
                STATE["cache"][fname] = []
            continue

        entries = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            img_w, img_h = result.orig_shape[1], result.orig_shape[0]

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().tolist()
                c = float(boxes.conf[i].cpu())
                cls_idx = int(boxes.cls[i].cpu()) if boxes.cls is not None else 0
                label = labels[cls_idx] if cls_idx < len(labels) else ""

                x1, y1 = max(0.0, xyxy[0]/img_w), max(0.0, xyxy[1]/img_h)
                x2, y2 = min(1.0, xyxy[2]/img_w), min(1.0, xyxy[3]/img_h)

                entries.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "conf": round(c, 4),
                    "is_auto": True,
                })
        STATE["cache"][fname] = entries

    save_cache(folder, STATE["cache"])
    STATE["inf_msg"] = "完了"
    STATE["is_inferencing"] = False


@app.route("/api/inference", methods=["POST"])
def inference():
    if STATE["is_inferencing"]:
        return jsonify({"error": "Already running"}), 400
        
    data = request.json
    label_entries = data.get("labels", [])
    conf = float(data.get("conf", 0.25))
    
    if not STATE["folder"]:
        return jsonify({"error": "No folder opened"}), 400
        
    for e in label_entries:
        if e["label"] not in STATE["label_list"]:
            STATE["label_list"].append(e["label"])
            
    threading.Thread(target=run_inference_thread, args=(STATE["folder"], label_entries, conf)).start()
    return jsonify({"success": True})


@app.route("/api/inference_status")
def inference_status():
    return jsonify({
        "is_inferencing": STATE["is_inferencing"],
        "progress": STATE["inf_progress"],
        "total": STATE["inf_total"],
        "msg": STATE["inf_msg"]
    })


@app.route("/api/export", methods=["POST"])
def export_dataset():
    data = request.json
    export_path = data.get("export_path")
    train_ratio = float(data.get("train_ratio", 0.8))
    
    if not STATE["folder"] or not STATE["images"]:
        return jsonify({"error": "画像フォルダが読み込まれていません"}), 400
    if not export_path:
        return jsonify({"error": "出力先フォルダを指定してください"}), 400
        
    valid_images = []
    for img_path in STATE["images"]:
        fname = Path(img_path).name
        has_bb = False
        if fname in STATE["cache"] and len(STATE["cache"][fname]) > 0:
            has_bb = True
        else:
            txt_path = Path(img_path).parent / (Path(img_path).stem + ".txt")
            if txt_path.exists() and txt_path.stat().st_size > 0:
                has_bb = True
        if has_bb:
            valid_images.append(img_path)

    if not valid_images:
        return jsonify({"error": "アノテーションが存在する画像が見つかりません"}), 400

    img_list = list(valid_images)
    random.seed(42)
    random.shuffle(img_list)

    split_idx = int(len(img_list) * train_ratio)
    if split_idx == len(img_list) and len(img_list) > 1: split_idx = len(img_list) - 1
    if split_idx == 0 and len(img_list) > 0: split_idx = 1

    train_imgs = img_list[:split_idx]
    val_imgs = img_list[split_idx:]

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
        return jsonify({"error": f"フォルダ作成失敗: {e}"}), 500

    def copy_data(imgs, img_dir, lbl_dir):
        copied = 0
        for img_path in imgs:
            p_img = Path(img_path)
            fname, stem = p_img.name, p_img.stem
            dest_img, dest_lbl = img_dir / fname, lbl_dir / f"{stem}.txt"
            
            try: shutil.copy2(img_path, dest_img)
            except: continue
            
            if fname in STATE["cache"]:
                save_yolo_from_entries(str(dest_img), STATE["cache"][fname], STATE["label_list"], out_dir=str(lbl_dir))
            else:
                src_txt = p_img.parent / f"{stem}.txt"
                if src_txt.exists(): shutil.copy2(src_txt, dest_lbl)
            copied += 1
        return copied

    train_count = copy_data(train_imgs, dirs["train_img"], dirs["train_lbl"])
    val_count = copy_data(val_imgs, dirs["val_img"], dirs["val_lbl"])

    yaml_path = Path(export_path) / "dataset.yaml"
    try:
        yaml_content = [
            f"path: {Path(export_path).absolute().as_posix()}",
            "train: train/images",
            "val: val/images",
            "",
            "names:"
        ]
        for idx, label in enumerate(STATE["label_list"]):
            yaml_content.append(f"  {idx}: {label}")
        yaml_path.write_text("\n".join(yaml_content), encoding="utf-8")
    except Exception as e:
        pass

    return jsonify({
        "success": True,
        "train_count": train_count,
        "val_count": val_count,
        "total": train_count + val_count
    })


def main():
    # Make templates dir
    os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)
    app.run(host="0.0.0.0", port=18000, debug=False)


if __name__ == "__main__":
    main()
