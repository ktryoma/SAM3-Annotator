# SAM3 Annotation Tool

SAM3（Segment Anything Model 3）の推論結果を初期値として活用し、YOLO形式のアノテーションデータを効率的に作成する PyQt6 デスクトップアプリです。

## 起動方法

```bat
run.bat
```
または

```bat
conda activate LCnet
python main.py
```

## 使い方

### 1. フォルダ選択
左ペインの「📂 フォルダを開く」で画像フォルダを選択します。

### 2. クラスラベル / 説明の設定

各ラベルには**ラベル名**と**説明 (プロンプト)** の2つを入力します。

| 項目 | 説明 |
|---|---|
| **ラベル名** | YOLOの出力クラス名（例: `chair`） |
| **説明 (プロンプト)** | SAM3のテキストプロンプト（例: `a wooden chair with four legs and a cushion`） |

- ラベル名を入力 → Tabキーで説明欄へ移動 → 説明を入力 → Enter または「+ 追加」
- テーブルをダブルクリックすると直接編集可能
- 説明文を詳細にするほど、SAM3の検出精度が向上します

### 3. SAM3 推論を実行
「▶ 推論を実行」ボタンを押すと、フォルダ内の全画像を一括推論します。
結果は `<画像フォルダ>/.sam3_cache/cache.json` にキャッシュされます。
次回以降はキャッシュから即時読み込みします。

### 4. アノテーション編集

| 操作 | 方法 |
|------|------|
| BBを選択 | クリック |
| ラベルを変更 | BBをダブルクリック |
| 新規BBを描く | 画像の空きエリアでドラッグ |
| BBを移動 | 選択してドラッグ |
| BBをリサイズ | 選択後、8つのハンドルをドラッグ |
| BBを削除 | 選択して `Delete` キー |
| ズーム | マウスホイール |
| パン（移動） | 中ボタンドラッグ |

### 5. 保存
- `💾 Save YOLO` ボタン、または `Ctrl+S` で現在の画像をYOLO形式で保存
- `◀ Prev` / `▶ Next` 移動時に未保存の場合は確認ダイアログが表示されます

### 出力フォーマット
```
<class_index> <x_center> <y_center> <width> <height>
```
値はすべて 0.0〜1.0 に正規化されています。

## ファイル構成

```
sam3ano/
├── main.py                  # エントリーポイント
├── run.bat                  # 起動スクリプト
├── app/
│   ├── annotation.py        # BoundingBox データクラス
│   ├── annotation_io.py     # JSON キャッシュ / YOLO txt 入出力
│   ├── inference_worker.py  # SAM3 推論バックグラウンドワーカー
│   ├── canvas.py            # 画像表示・BB描画・インタラクション
│   ├── label_dialog.py      # ラベル選択ダイアログ
│   ├── main_window.py       # メインウィンドウ
│   └── panels/
│       ├── left_panel.py    # フォルダ選択・ラベル管理・推論
│       └── right_panel.py   # BB リスト表示
├── assets/
│   └── style.qss            # ダークテーマ
├── model/
│   └── sam3.pt              # SAM3 モデルウェイト
└── data/                    # サンプル画像
```

## 必要な環境

- conda 環境: `LCnet`
  - PyQt6
  - ultralytics >= 8.3
  - opencv-python
  - numpy
