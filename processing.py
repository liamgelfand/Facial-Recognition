"""
Preprocessing pipeline — supports two dataset formats:

FORMAT A  (folder structure, FER-2013 original labels)
  dataset/train/[emotion]/image.png  ←  what you already have
  Run as-is. Accuracy ceiling ~65-70% with transfer learning.

FORMAT B  (CSV + FER+ labels) — RECOMMENDED for best accuracy (~80%+)
  Requires two files in the project root:
    fer2013.csv      — download from Kaggle (the original CSV version of FER-2013)
                       https://www.kaggle.com/datasets/deadskull7/fer2013
    fer2013new.csv   — Microsoft FER+ relabeling (10 annotators per image)
                       https://github.com/microsoft/FERPlus/blob/master/data/fer2013new.csv
  FER+ dramatically improves label quality, which is the main bottleneck on FER-2013.

Output: processed_dataset/{train,test}_{images,labels}.npy
  Images: float32, shape (N, 96, 96, 3), range [0, 1]  (RGB)
"""

import os
import cv2
import numpy as np

IMAGE_SIZE = (96, 96)  # MobileNetV2 minimum input size

# FER+ vote columns → label name (first 6; skip contempt/unknown/NF)
FERPLUS_COLS = ["neutral", "happiness", "surprise", "fear", "angry", "disgust"]
FERPLUS_LABEL_MAP = {
    "neutral":   "neutral",
    "happiness": "happy",
    "surprise":  "surprise",
    "fear":      "fear",
    "angry":     "angry",
    "disgust":   "disgust",
}


# ── Format B: original FER-2013 CSV + FER+ labels ─────────────────────────────

def load_from_csv_with_ferplus(fer2013_csv: str, ferplus_csv: str, usage: str):
    """
    Load images from the original FER-2013 pixel CSV and apply FER+ majority-vote
    labels. Rows in both CSVs are in the same order within each split, so we join
    by position.
    """
    import pandas as pd

    df_orig = pd.read_csv(fer2013_csv)
    df_fp   = pd.read_csv(
        ferplus_csv, header=None,
        names=["usage", "image",
               "neutral", "happiness", "surprise", "fear",
               "angry", "disgust", "contempt", "unknown", "NF"],
    )

    df_orig = df_orig[df_orig["Usage"] == usage].reset_index(drop=True)
    df_fp   = df_fp[df_fp["usage"] == usage].reset_index(drop=True)

    images, labels = [], []
    skipped = 0

    for i in range(min(len(df_orig), len(df_fp))):
        row_fp = df_fp.iloc[i]

        # Skip images with too many "unknown" or "NF" votes
        if row_fp["unknown"] + row_fp["NF"] > 5:
            skipped += 1
            continue

        # Majority vote across first 6 emotions
        votes        = [row_fp[c] for c in FERPLUS_COLS]
        dominant_idx = int(np.argmax(votes))
        label        = FERPLUS_LABEL_MAP[FERPLUS_COLS[dominant_idx]]

        # Reconstruct grayscale image from pixel string, then convert to RGB
        pixels = (
            np.array(df_orig.iloc[i]["pixels"].split(), dtype=np.uint8)
            .reshape(48, 48)
        )
        img_rgb = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
        img_rgb = cv2.resize(img_rgb, IMAGE_SIZE)

        images.append(img_rgb)
        labels.append(label)

    print(f"  {usage}: loaded {len(images)}, skipped {skipped}")
    return np.array(images, dtype=np.float32), np.array(labels)


# ── Format A: folder structure ─────────────────────────────────────────────────

def load_from_folders(directory: str):
    """Load images from dataset/[emotion]/ folder layout."""
    images, labels = [], []
    for emotion in sorted(os.listdir(directory)):
        folder = os.path.join(directory, emotion)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img   = cv2.imread(fpath)          # reads as BGR colour
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            labels.append(emotion)
    return np.array(images, dtype=np.float32), np.array(labels)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    train_dir    = "dataset/train"
    test_dir     = "dataset/test"
    save_dir     = "processed_dataset"
    fer2013_csv  = "fer2013.csv"
    ferplus_csv  = "fer2013new.csv"

    os.makedirs(save_dir, exist_ok=True)

    use_ferplus = os.path.isfile(fer2013_csv) and os.path.isfile(ferplus_csv)

    if use_ferplus:
        print("FER+ mode: using fer2013.csv + fer2013new.csv (cleaner labels)")
        print("Loading training split ...")
        train_images, train_labels = load_from_csv_with_ferplus(
            fer2013_csv, ferplus_csv, "Training"
        )
        print("Loading test split ...")
        test_images, test_labels = load_from_csv_with_ferplus(
            fer2013_csv, ferplus_csv, "PublicTest"
        )
    else:
        print("Folder mode: using dataset/ folder structure (FER-2013 original labels)")
        if not os.path.isfile(fer2013_csv):
            print("  Tip: for +10-15% accuracy, download fer2013.csv + fer2013new.csv")
            print("  See README.md for links.")
        print("Loading training set ...")
        train_images, train_labels = load_from_folders(train_dir)
        print("Loading test set ...")
        test_images, test_labels = load_from_folders(test_dir)

    # Normalise to [0, 1]
    train_images /= 255.0
    test_images  /= 255.0

    print(f"\nTrain: {len(train_images)} images | Test: {len(test_images)} images")
    print(f"Image shape: {train_images[0].shape}")

    np.save(os.path.join(save_dir, "train_images.npy"), train_images)
    np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(save_dir, "test_images.npy"),  test_images)
    np.save(os.path.join(save_dir, "test_labels.npy"),  test_labels)
    print("Saved to processed_dataset/")


if __name__ == "__main__":
    main()
