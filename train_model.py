"""
Transfer learning pipeline using MobileNetV2 pretrained on ImageNet.

Training runs in two phases:
  Phase 1 — Feature extraction
    The MobileNetV2 base is fully frozen. Only the custom classification
    head is trained. This converges quickly and avoids overwriting the
    pretrained features with random gradients.

  Phase 2 — Fine-tuning
    The top layers of the base are unfrozen and trained jointly with the
    head at a very low learning rate. This adapts the high-level features
    to the emotion domain while preserving low-level edge/texture filters.

Run processing.py first to generate processed_dataset/*.npy.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE     = 96
BATCH_SIZE     = 32
PHASE1_EPOCHS  = 20   # head-only training (fast convergence)
PHASE2_EPOCHS  = 40   # fine-tuning (slower, lower LR)
UNFREEZE_LAYERS = 40  # how many layers from the top of the base to unfreeze


# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading data ...")
train_images = np.load("processed_dataset/train_images.npy")
train_labels = np.load("processed_dataset/train_labels.npy")
test_images  = np.load("processed_dataset/test_images.npy")
test_labels  = np.load("processed_dataset/test_labels.npy")

# Backward-compat: if saved as grayscale (H, W) or (H, W, 1), convert to RGB
if train_images.ndim == 3:
    train_images = np.stack([train_images] * 3, axis=-1)
    test_images  = np.stack([test_images]  * 3, axis=-1)
elif train_images.shape[-1] == 1:
    train_images = np.concatenate([train_images] * 3, axis=-1)
    test_images  = np.concatenate([test_images]  * 3, axis=-1)

# MobileNetV2 preprocessing: [0,1] → [-1, 1]
train_images = tf.keras.applications.mobilenet_v2.preprocess_input(
    train_images * 255.0
)
test_images = tf.keras.applications.mobilenet_v2.preprocess_input(
    test_images * 255.0
)


# ── Labels ────────────────────────────────────────────────────────────────────

le = LabelEncoder()
train_enc = le.fit_transform(train_labels)
test_enc  = le.transform(test_labels)
num_classes = len(le.classes_)
print("Classes:", dict(enumerate(le.classes_)))

class_weights = compute_class_weight(
    "balanced", classes=np.arange(num_classes), y=train_enc
)
class_weight_dict = dict(enumerate(class_weights))


# ── Model ─────────────────────────────────────────────────────────────────────

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False  # frozen for phase 1

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="EmotionMobileNetV2")
model.summary()


# ── Phase 1: train head only ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 1 — Training classification head (base frozen)")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_p1 = [
    tf.keras.callbacks.ModelCheckpoint(
        "emotion_recognition_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
]

history_p1 = model.fit(
    train_images, train_enc,
    epochs=PHASE1_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(test_images, test_enc),
    class_weight=class_weight_dict,
    callbacks=callbacks_p1,
)


# ── Phase 2: fine-tune top layers ─────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"PHASE 2 — Fine-tuning top {UNFREEZE_LAYERS} layers of MobileNetV2")
print("=" * 60)

base.trainable = True
for layer in base.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable_count}")

# Very low LR to avoid destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    tf.keras.callbacks.ModelCheckpoint(
        "emotion_recognition_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
]

history_p2 = model.fit(
    train_images, train_enc,
    epochs=PHASE2_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(test_images, test_enc),
    class_weight=class_weight_dict,
    callbacks=callbacks_p2,
)


# ── Save history & evaluate ───────────────────────────────────────────────────

combined = {
    k: history_p1.history[k] + history_p2.history.get(k, [])
    for k in history_p1.history
}
np.save("training_history.npy", combined)
print("Training history saved → training_history.npy")

test_loss, test_acc = model.evaluate(test_images, test_enc, verbose=2)
print(f"\nFinal test accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print(f"Final test loss     : {test_loss:.4f}")
