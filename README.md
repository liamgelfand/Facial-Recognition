# Facial Emotion Recognition

A real-time facial emotion recognition system using **MobileNetV2 transfer learning** fine-tuned on FER-2013 with optional FER+ label cleaning. Detects seven emotions from a live webcam feed with bounding boxes, confidence labels, and per-emotion probability bars.

---

## Demo

```
python realtime_demo.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `p` | Pause / resume |

The demo uses OpenCV's Haar cascade to locate faces, then feeds each 96×96 RGB crop through the fine-tuned MobileNetV2 model in real time.

---

## Emotions

7 classes from FER-2013 / FER+:

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## Architecture

MobileNetV2 pretrained on ImageNet as a frozen feature extractor, with a custom classification head. Training runs in two phases:

| Phase | Base | Head | LR |
|-------|------|------|----|
| 1 — Feature extraction | Frozen | Trained | 1e-3 |
| 2 — Fine-tuning | Top 40 layers unfrozen | Trained | 1e-5 |

**Head:**  `GlobalAveragePooling2D → Dense(256, ReLU) → BatchNorm → Dropout(0.5) → Dense(7, Softmax)`

**Input:** 96 × 96 × 3 (RGB)

---

## Accuracy

| Approach | Val Accuracy |
|----------|-------------|
| Custom CNN, no augmentation (original) | ~45–50% |
| Custom CNN + augmentation + class weights | ~56–62% |
| **MobileNetV2 fine-tuned (FER-2013 labels)** | **~65–70%** |
| **MobileNetV2 fine-tuned (FER+ labels)** | **~78–83%** |
| Human performance on FER-2013 | ~65–70% |
| State-of-the-art (ensemble + attention) | ~73–75% |

> FER-2013 labels were crowdsourced with a single annotator per image. FER+ re-labels every image with 10 annotators, dramatically reducing noise.

---

## Dataset Setup

### Option A — Folder format (quick start, FER-2013 labels)

Download from Kaggle: [msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Place as:
```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── ...
```

### Option B — CSV format + FER+ labels (recommended, ~80%+ accuracy)

1. Download the original `fer2013.csv` from Kaggle:
   [deadskull7/fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013)

2. Download `fer2013new.csv` from Microsoft's FER+ repo:
   [microsoft/FERPlus → data/fer2013new.csv](https://github.com/microsoft/FERPlus/blob/master/data/fer2013new.csv)

3. Place both files in the project root:
   ```
   facial_recognition/
   ├── fer2013.csv
   ├── fer2013new.csv
   └── ...
   ```

`processing.py` automatically detects which format is available.

---

## Setup

```bash
git clone <your-repo-url>
cd facial_recognition

python -m venv facial-recognition
.\facial-recognition\Scripts\activate      # Windows
# source facial-recognition/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

---

## Usage

### 1 · Preprocess

```bash
python processing.py
```

Saves 96×96 RGB float arrays to `processed_dataset/`.

### 2 · Train

```bash
python train_model.py
```

Phase 1 trains the head only (~10–15 min on CPU). Phase 2 fine-tunes the top MobileNetV2 layers. Best weights saved automatically to `emotion_recognition_model.keras`.

> **Tip:** Use Google Colab with a free T4 GPU — total training time drops to ~15–30 minutes.

### 3 · Evaluate

```bash
python evaluate.py
```

Outputs: `confusion_matrix.png`, `per_class_accuracy.png`, `class_distribution.png`, `training_curves.png`.

### 4 · Real-time demo

```bash
python realtime_demo.py
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | MobileNetV2 backbone, training, inference |
| OpenCV | Webcam capture, Haar cascade face detection, frame rendering |
| NumPy | Array operations, data serialisation |
| scikit-learn | Label encoding, class-weight computation, evaluation metrics |
| Matplotlib / Seaborn | Training curves, confusion matrix, distribution plots |
| pandas | FER+ CSV parsing (Format B only) |

---

## Project Structure

```
facial_recognition/
├── dataset/                          # FER-2013 images (gitignored)
├── processed_dataset/                # Preprocessed .npy arrays (gitignored)
├── fer2013.csv                       # Optional — original CSV format
├── fer2013new.csv                    # Optional — FER+ improved labels
├── processing.py                     # Image loading, normalisation, FER+ support
├── train_model.py                    # MobileNetV2 transfer learning pipeline
├── evaluate.py                       # Evaluation plots & classification report
├── realtime_demo.py                  # Live webcam emotion detection
├── emotion_recognition_model.keras   # Trained model weights
├── training_history.npy              # Loss/accuracy curves
└── requirements.txt
```
