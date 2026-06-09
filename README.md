# SAM3 Annotation Tool

A premium PyQt6 desktop application designed to streamline the creation of YOLO-format annotation datasets by utilizing Segment Anything Model 3 (SAM3) inference results as initial bounding boxes. It features a sleek custom dark mode, a bilingual user interface, and rich dataset management tools.

---

## ✨ Key Features

- **SAM3-Powered Auto-Labeling**: Run batch inference on an entire folder of images using fine-grained natural language prompts to auto-generate high-quality initial bounding boxes.
- **Bilingual Interface (Japanese / English)**: Toggle between English and Japanese with a single click. Tooltips, panels, dialogs, and messages are fully localized.
- **Index-Based Navigation (Jump to Image)**: Instantly jump to any image in your dataset by typing or scrolling to its index number in the navigation spinbox.
- **"Detected Only" Filtering**: Filter your image list with one click to show only images containing at least one bounding box (annotated either via SAM3 auto-inference or manual edits), facilitating rapid validation and refinement.
- **YOLO Dataset Export**: Seamlessly partition your annotated dataset into `train` and `val` splits at a custom ratio (e.g., 80% / 20%). The tool automatically copies only the annotated images, writes the corresponding YOLO txt label files (incorporating any unsaved in-memory edits), and outputs a ready-to-train `dataset.yaml` file.
- **Local JSON Caching**: Inference results are cached locally (`.sam3_cache/cache.json`) to allow instant loading on subsequent launches, bypassing heavy model executions.
- **Advanced Interactive Editing & Cancel Edits**: Drag to draw new boxes, select, move, resize (using 8 interactive handles), and delete boxes. Click **"Cancel Edits"** at any time to discard unsaved changes and revert the canvas to the last saved snapshot.
- **Batch Saving**: Save current changes via **"Save"** (`Ctrl+S`), or export all in-memory cache annotations to their respective YOLO txt files in one click using **"Save All"** (`Ctrl+Shift+S`).

---

## 🚀 Getting Started / Running the App

### Environment
Please prepare the environment using vnev or conda.
You can refer the required packages from Environment Requirements section.

### Model file
You need to prepare the model file (sam3.pt) and place it in the model directory. You can download it from the [Hugging Face website](https://huggingface.co/bodhicitta/sam3/blob/main/sam3.pt).

### How to run
#### Desktop Application
Activate your Python environment and launch the desktop application:

```bat
python main.py
```

#### Web Application (Recommend)
You can also run the web application, not depending on the OS. 
First, you need to get docker image from Docker Hub as follows:

```bat
docker pull ktryoma/sam3-annotation:v1
```

Then, run the docker compose command to start the web server:

```bat
docker compose up
```
After the server starts, open your web browser and navigate to `http://localhost:18000` to access the annotation tool.

## 💡 How to Use

### 1. Open Folder
Click **"📂 Open Folder"** (📂 フォルダを開く) in the left panel and select your image directory.

### 2. Configure Class Labels & Descriptions (Prompts)
In the class label table, define the target classes you wish to detect. Each class entry requires:

| Field | Description | Example |
| :--- | :--- | :--- |
| **Label** (ラベル) | The class name output to YOLO txt files | `chair` |
| **Description (Prompt)** (説明/プロンプト) | A detailed natural language description used by SAM3 | `a wooden chair with four legs and a cushion` |

*Tips:*
- Enter the label $\rightarrow$ Press `Tab` to move to the description field $\rightarrow$ Enter the prompt $\rightarrow$ Press `Enter` or click **"+ Add"** to insert.
- Double-click any cell in the table to edit its content directly.
- The more descriptive and detailed the prompt, the higher the SAM3 detection accuracy.

### 3. Execute SAM3 Auto-Inference
Adjust the detection confidence threshold (e.g., `0.25`) and click **"▶ Run Inference"** (▶ 推論を実行).
- A background worker will process the entire folder.
- Results are saved under `<folder>/.sam3_cache/cache.json` for lightning-fast loads.
- A summary of successfully detected counts per label will display upon completion.

### 4. Edit and Fine-Tune Annotations
Click on images in the canvas to inspect, draw, or edit bounding boxes.

| Action / Operation | Control / Shortcut |
| :--- | :--- |
| **Select Bounding Box** | Left-click on the box |
| **Draw New Bounding Box** | Drag left-click in an empty area |
| **Move Bounding Box** | Drag the selected box |
| **Resize Bounding Box** | Drag any of the 8 interactive handles around a selected box |
| **Change Label of Box** | Double-click the box, or double-click its entry in the right panel |
| **Delete Bounding Box** | Select a box and press the `Delete` key |
| **Zoom In / Out** | Mouse wheel |
| **Pan Canvas** | Click and drag with the middle mouse button (scroll wheel) |
| **Go to Previous Image** | Click `◀ Prev` or press the `Left Arrow` key |
| **Go to Next Image** | Click `Next ▶` or press the `Right Arrow` key |
| **Jump to Specific Index** | Enter or scroll to the target number in the **"Jump"** (ジャンプ) spinbox |
| **Revert / Cancel Edits** | Click **"↩ Cancel Edits"** (↩ 編集キャンセル) to discard current unsaved edits |

### 5. Filter "Detected Only"
Check **"Detected Only"** (検出あり画像のみ) in the navigation bar.
- This immediately filters your active list and updates navigation bounds to only include images containing one or more annotations.
- Excellent for jumping directly between valid predictions to verify or correct them.

### 6. Save Annotations
- **Save Current**: Click **"💾 Save"** or press `Ctrl+S` to write YOLO txt files for the current image.
- **Save All**: Click **"💾 Save All"** or press `Ctrl+Shift+S` to save YOLO files for all annotated images in the folder.

**YOLO Output Format:**
```
<class_index> <x_center> <y_center> <width> <height>
```
*All values are normalized between `0.0` and `1.0`.*

### 7. Export YOLO Dataset
Click **"📦 Export Dataset"** (📦 学習データを出力) in the left panel to configure your export:
1. **Output Directory**: Browse and select a destination folder.
2. **Train/Val Split**: Adjust the training split percentage (e.g., `80%` Train / `20%` Val).
3. **Execution**:
   - The application automatically extracts only the images that have at least one valid annotation.
   - Images are randomly shuffled and partitioned based on the split ratio.
   - Corresponding images and YOLO-format txt files (including any current unsaved in-memory edits) are copied and exported.
   - A `dataset.yaml` defining categories and absolute train/val paths is automatically created in the root of the output directory.
   - The exported folder is fully configured and ready for immediate YOLO training.

---

## 📁 Directory Structure

```
SAN3-Annotator/
├── main.py                  # Application entry point
├── run.bat                  # Windows startup batch script
├── app/
│   ├── annotation.py        # BoundingBox data structures
│   ├── annotation_io.py     # JSON cache and YOLO txt import/export logic
│   ├── inference_worker.py  # Background worker thread for SAM3 inference
│   ├── canvas.py            # Custom interactive QGraphicsView for bounding box rendering/editing
│   ├── label_dialog.py      # Popup dialog for selecting and adding labels
│   ├── export_dialog.py     # YOLO dataset partitioning & export setup dialog
│   ├── main_window.py       # Core window management & orchestration
│   └── panels/
│       ├── left_panel.py    # Left sidebar for folder selection, label setup & inference execution
│       └── right_panel.py   # Right sidebar for listing bounding boxes and toggling UI language
├── assets/
│   └── style.qss            # Custom CSS dark theme stylesheet
├── model/
│   └── sam3.pt              # Segment Anything Model 3 weights (Ultralytics)
└── data/                    # Directory for sample images or test datasets
```

---

## 💻 Environment Requirements

- **Requirement.txt**: it will be uploaded soon... 
- **Core Dependencies**:
  - `PyQt6` (Desktop UI Framework)
  - `ultralytics >= 8.3` (Inference engine for SAM3 & YOLO)
  - `opencv-python`
  - `numpy`
