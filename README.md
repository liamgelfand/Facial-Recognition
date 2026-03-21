# Facial Emotion Recognition

A real-time facial emotion recognition system built with a custom CNN trained on the **FER-2013** dataset. The model classifies seven emotions from a live webcam feed — drawing bounding boxes, emotion labels, and per-emotion confidence bars directly on the frame.

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

The demo uses OpenCV's Haar cascade detector to locate faces, then feeds each 48×48 grayscale crop through the trained CNN in real time.

---

## Emotions

The model recognises **7 classes** drawn from FER-2013:

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## Architecture

A four-block CNN with Batch Normalisation and Global Average Pooling:

| Block | Layers | Filters |
|-------|--------|---------|
| 1 | Conv → BN → ReLU × 2, MaxPool, Dropout | 32 |
| 2 | Conv → BN → ReLU × 2, MaxPool, Dropout | 64 |
| 3 | Conv → BN → ReLU × 2, MaxPool, Dropout | 128 |
| 4 | Conv → BN → ReLU, MaxPool, Dropout | 256 |
| Head | GlobalAvgPool → Dense(512) → BN → Dropout → Dense(7, softmax) | — |

- **Input:** 48 × 48 × 1 (grayscale)
- **Optimizer:** Adam with `ReduceLROnPlateau`
- **Loss:** Sparse Categorical Cross-Entropy
- **Regularisation:** Dropout, Batch Normalisation, data augmentation, class weights

---

## Training improvements over baseline

| Feature | Baseline | This version |
|---------|----------|--------------|
| Conv blocks | 3 | 4 |
| Batch Normalisation | ✗ | ✓ |
| Data augmentation | ✗ | flip · rotate · zoom · translate |
| Class-imbalance weighting | ✗ | ✓ (`sklearn.compute_class_weight`) |
| Learning rate scheduling | ✗ | `ReduceLROnPlateau` |
| Early stopping | ✗ | ✓ (patience = 10) |
| Best-model checkpoint | ✗ | ✓ |
| Training history saved | ✗ | `training_history.npy` |

---

## Dataset — FER-2013

[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) contains ~35 000 grayscale 48×48 face images labelled with one of seven emotions. It is available on Kaggle (free account required).

Download and place it as:

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

---

## Setup

```bash
# Clone repo
git clone https://github.com/liamgelfand/Facial-Recognition
cd facial_recognition

# Create & activate virtual environment
python -m venv facial-recognition
.\facial-recognition\Scripts\activate      # Windows
# source facial-recognition/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1 · Preprocess

```bash
python processing.py
```

Creates `processed_dataset/` with four `.npy` arrays.

### 2 · Train

```bash
python train_model.py
```

Trains for up to 50 epochs with early stopping. Saves the best weights to `emotion_recognition_model.keras` and training history to `training_history.npy`.

### 3 · Evaluate

```bash
python evaluate.py
```

Generates:
- `confusion_matrix.png` — raw and normalised confusion matrices
- `per_class_accuracy.png` — bar chart of per-class accuracy
- `class_distribution.png` — train/test sample counts per emotion
- `training_curves.png` — accuracy and loss over epochs

### 4 · Real-time demo

```bash
python realtime_demo.py
```

Requires a working webcam.

---

## Tech stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | Model definition, training, inference |
| OpenCV | Webcam capture, face detection (Haar cascade), frame rendering |
| NumPy | Array operations, data serialisation |
| scikit-learn | Label encoding, class-weight computation, evaluation metrics |
| Matplotlib / Seaborn | Training curves, confusion matrix, distribution plots |

---

## Project structure

```
facial_recognition/
├── dataset/                    # FER-2013 images (gitignored)
├── processed_dataset/          # Preprocessed .npy arrays (gitignored)
├── processing.py               # Image loading & normalisation
├── train_model.py              # CNN training pipeline
├── evaluate.py                 # Evaluation plots & classification report
├── realtime_demo.py            # Live webcam emotion detection
├── emotion_recognition_model.keras
├── training_history.npy
└── requirements.txt
```
