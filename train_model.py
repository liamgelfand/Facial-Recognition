"""
Train a CNN for facial emotion recognition on the FER-2013 dataset.

Run processing.py first to generate processed_dataset/*.npy, then:
    python train_model.py

Outputs
-------
emotion_recognition_model.keras   — best model weights (saved by ModelCheckpoint)
training_history.npy               — dict with loss / accuracy curves for evaluate.py
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# ── Load data ──────────────────────────────────────────────────────────────────

train_images = np.load("processed_dataset/train_images.npy")
train_labels = np.load("processed_dataset/train_labels.npy")
test_images  = np.load("processed_dataset/test_images.npy")
test_labels  = np.load("processed_dataset/test_labels.npy")

# Encode string labels → integers (LabelEncoder sorts alphabetically)
label_encoder = LabelEncoder()
train_labels  = label_encoder.fit_transform(train_labels)
test_labels   = label_encoder.transform(test_labels)
print("Label mapping:", dict(enumerate(label_encoder.classes_)))

# Add channel dimension: (N, 48, 48) → (N, 48, 48, 1)
train_images = train_images.reshape(-1, 48, 48, 1).astype("float32")
test_images  = test_images.reshape(-1, 48, 48, 1).astype("float32")

num_classes = len(label_encoder.classes_)


# ── Class weights (handle FER-2013 imbalance — disgust is underrepresented) ───

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels,
)
class_weight_dict = dict(enumerate(class_weights_array))
print("Class weights:", {label_encoder.classes_[k]: f"{v:.2f}"
                          for k, v in class_weight_dict.items()})


# ── Data augmentation ─────────────────────────────────────────────────────────

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.06, 0.06),
], name="augmentation")


# ── Model architecture ────────────────────────────────────────────────────────
# Four conv blocks with Batch Normalisation, then two Dense layers.
# BatchNorm stabilises training and lets us train deeper without overfitting.

def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = data_augmentation(inputs)

    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 4
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="EmotionCNN")


model = build_model(num_classes)
model.summary()


# ── Compile ───────────────────────────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

callbacks = [
    # Save only the best epoch (by val_accuracy)
    tf.keras.callbacks.ModelCheckpoint(
        filepath="emotion_recognition_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    # Halve LR when val_loss stalls for 3 epochs
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
    # Stop early if val_accuracy hasn't improved for 10 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ── Train ─────────────────────────────────────────────────────────────────────

history = model.fit(
    train_images,
    train_labels,
    epochs=50,
    batch_size=64,
    validation_data=(test_images, test_labels),
    class_weight=class_weight_dict,
    callbacks=callbacks,
)

# Persist history so evaluate.py can plot training curves without retraining
np.save("training_history.npy", history.history)
print("Training history saved → training_history.npy")


# ── Final evaluation ──────────────────────────────────────────────────────────

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Test loss     : {test_loss:.4f}")
