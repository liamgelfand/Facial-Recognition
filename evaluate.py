"""
Model evaluation — generates four plots and a per-class report.

Usage
-----
    python evaluate.py                    # uses defaults below
    python evaluate.py --history none     # skip training-curve plots

Outputs
-------
  confusion_matrix.png
  training_curves.png   (only if training_history.npy exists)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "emotion_recognition_model.keras"
HISTORY_PATH = "training_history.npy"
DATA_DIR     = "processed_dataset"
EMOTIONS     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor":   "#1e1e2e",
    "axes.edgecolor":   "#585878",
    "axes.labelcolor":  "#cdd6f4",
    "xtick.color":      "#cdd6f4",
    "ytick.color":      "#cdd6f4",
    "text.color":       "#cdd6f4",
    "grid.color":       "#313244",
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
})

PALETTE = ["#f38ba8", "#fab387", "#f9e2af",
           "#a6e3a1", "#94e2d5", "#89dceb", "#cba6f7"]


# ── Load data & model ─────────────────────────────────────────────────────────

def load_test_data():
    test_images = np.load(f"{DATA_DIR}/test_images.npy").reshape(-1, 48, 48, 1).astype("float32")
    raw_labels  = np.load(f"{DATA_DIR}/test_labels.npy")
    le = LabelEncoder()
    test_labels = le.fit_transform(raw_labels)
    return test_images, test_labels, le


def load_train_data():
    train_images = np.load(f"{DATA_DIR}/train_images.npy").reshape(-1, 48, 48, 1).astype("float32")
    raw_labels   = np.load(f"{DATA_DIR}/train_labels.npy")
    le = LabelEncoder()
    train_labels = le.fit_transform(raw_labels)
    return train_images, train_labels


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix — Facial Emotion Recognition",
                 fontsize=15, fontweight="bold", color="#cdd6f4", y=1.02)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Normalised (row %)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=class_names, yticklabels=class_names,
            cmap="mako", ax=ax, linewidths=0.4,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


def plot_per_class_accuracy(y_true, y_pred, class_names,
                             save_path="per_class_accuracy.png"):
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(class_names, per_class * 100, color=PALETTE, edgecolor="none",
                  width=0.6, zorder=3)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Test Accuracy", fontsize=13, fontweight="bold")
    ax.axhline(np.mean(per_class) * 100, color="#f5c2e7", linestyle="--",
               linewidth=1.2, label=f"Mean: {np.mean(per_class)*100:.1f}%")
    ax.legend(fontsize=10)
    ax.grid(axis="y", zorder=0)
    ax.tick_params(axis="x", rotation=20)

    for bar, acc in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc*100:.1f}%",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


def plot_training_curves(history: dict, save_path="training_curves.png"):
    epochs = range(1, len(history["accuracy"]) + 1)

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Accuracy
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, history["accuracy"],     color="#a6e3a1", lw=2,   label="Train")
    ax1.plot(epochs, history["val_accuracy"], color="#89dceb", lw=2,   label="Validation",
             linestyle="--")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, history["loss"],     color="#f38ba8", lw=2,   label="Train")
    ax2.plot(epochs, history["val_loss"], color="#fab387", lw=2,   label="Validation",
             linestyle="--")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


def plot_class_distribution(save_path="class_distribution.png"):
    """Bar chart of sample counts per emotion (train vs test)."""
    import os
    train_dir = "dataset/train"
    test_dir  = "dataset/test"

    if not os.path.isdir(train_dir):
        print("dataset/ not found, skipping class-distribution plot.")
        return

    train_counts = {e: 0 for e in EMOTIONS}
    test_counts  = {e: 0 for e in EMOTIONS}

    for emotion in EMOTIONS:
        tp = os.path.join(train_dir, emotion)
        if os.path.isdir(tp):
            train_counts[emotion] = len(os.listdir(tp))
        vp = os.path.join(test_dir, emotion)
        if os.path.isdir(vp):
            test_counts[emotion] = len(os.listdir(vp))

    x   = np.arange(len(EMOTIONS))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, [train_counts[e] for e in EMOTIONS],
           width=w, label="Train", color=PALETTE, alpha=0.85, edgecolor="none")
    ax.bar(x + w / 2, [test_counts[e]  for e in EMOTIONS],
           width=w, label="Test",  color=PALETTE, alpha=0.45, edgecolor="none",
           hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=20)
    ax.set_ylabel("Image count")
    ax.set_title("Class Distribution (FER-2013)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=MODEL_PATH,   help="Path to .keras model")
    parser.add_argument("--history", default=HISTORY_PATH, help="Path to training_history.npy (or 'none')")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    model.summary()

    print("Loading test data …")
    test_images, test_labels, le = load_test_data()
    class_names = list(le.classes_)

    print("Predicting …")
    pred_probs = model.predict(test_images, batch_size=128, verbose=1)
    pred_labels = np.argmax(pred_probs, axis=1)

    # ── Text report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(test_labels, pred_labels, target_names=class_names))

    overall_acc = np.mean(pred_labels == test_labels)
    print(f"Overall Test Accuracy: {overall_acc * 100:.2f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(test_labels, pred_labels, class_names)
    plot_per_class_accuracy(test_labels, pred_labels, class_names)
    plot_class_distribution()

    if args.history.lower() != "none":
        import os
        hist_path = args.history
        if os.path.isfile(hist_path):
            history = np.load(hist_path, allow_pickle=True).item()
            plot_training_curves(history)
        else:
            print(f"History file not found ({hist_path}), skipping training curves.")

    print("\nAll plots saved to the current directory.")


if __name__ == "__main__":
    main()
