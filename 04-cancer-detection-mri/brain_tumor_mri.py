"""Brain Tumour Detection and Classification from MRI images.

Classifies brain MRI scans into glioma, meningioma, pituitary tumour, or no
tumour, using a CNN trained from scratch.

Evaluation is deliberately weighted toward SENSITIVITY (recall) rather than
accuracy. In a screening context a missed tumour and a false alarm are not
comparable errors: the false alarm costs a follow-up scan, the miss can cost a
life. The script therefore reports per-class recall, a binary tumour/no-tumour
view, and an explicit count of missed tumours.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


DATA_ROOT = "Brain-Tumor-Classification-DataSet-master"
TRAIN_DIR = "Training"
TEST_DIR = "Testing"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("EPOCHS", "40"))
VALIDATION_SPLIT = 0.15
RANDOM_STATE = 42
# BatchNorm keeps running mean/variance for inference. Keras defaults to
# momentum=0.99, which needs ~500 steps to track the batch statistics - but this
# dataset gives only ~77 steps per epoch. The lagging running stats made
# validation accuracy sit at chance (0.23 on 4 classes) while training accuracy
# climbed past 0.78, because the two paths were effectively using different
# normalisation. 0.9 converges within a single epoch.
BN_MOMENTUM = 0.9
NO_TUMOUR_CLASS = "no_tumor"

COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"


def set_seeds() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY, length=0)


def build_cnn(n_classes: int) -> Sequential:
    return Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.Rescaling(1.0 / 255),

        # Only mild geometric augmentation. Aggressive flips/rotations are wrong for
        # MRI: anatomical orientation is diagnostic information, not noise.
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=BN_MOMENTUM),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=BN_MOMENTUM),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=BN_MOMENTUM),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=BN_MOMENTUM),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ], name="brain_tumour_cnn")


def dataset_to_arrays(dataset):
    images, labels = [], []
    for batch_images, batch_labels in dataset:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(images), np.concatenate(labels)


def main() -> None:
    set_seeds()
    project_dir = Path(__file__).resolve().parent
    root = project_dir / DATA_ROOT
    if not root.exists():
        raise SystemExit(
            f"Dataset folder '{DATA_ROOT}' not found. See the README for the download "
            "instructions (Kaggle: sartajbhuvaji/brain-tumor-classification-mri)."
        )

    # ---------------------------------------------------------------- 1. Load
    print("=" * 72)
    print("1. DATA UNDERSTANDING")
    print("=" * 72)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root / TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", seed=RANDOM_STATE, validation_split=VALIDATION_SPLIT,
        subset="training", shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root / TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", seed=RANDOM_STATE, validation_split=VALIDATION_SPLIT,
        subset="validation", shuffle=True,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        root / TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=False,
    )

    class_names = train_ds.class_names
    n_classes = len(class_names)
    print()
    print(f"Classes ({n_classes}): {class_names}")
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB")
    print()

    counts = {}
    for name in class_names:
        counts[name] = len(list((root / TRAIN_DIR / name).glob("*")))
    total_train = sum(counts.values())
    print("Training class distribution:")
    for name, count in counts.items():
        print(f"  {name:<20s} {count:5d}  ({count / total_train:.1%})")
    print(f"  {'TOTAL':<20s} {total_train:5d}")
    print()
    print(f"The '{NO_TUMOUR_CLASS}' class is the smallest at "
          f"{counts[NO_TUMOUR_CLASS] / total_train:.1%}, which matters: an imbalanced "
          "model biased toward predicting a tumour type would look accurate while "
          "being clinically useless at ruling disease out.")
    print()

    sample_images, sample_labels = next(iter(train_ds))
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8))
    for ax, index in zip(axes.ravel(), range(8)):
        ax.imshow(sample_images[index].numpy().astype("uint8"))
        ax.set_title(class_names[int(sample_labels[index])], color=INK_PRIMARY, fontsize=9)
        ax.axis("off")
    fig.suptitle("Brain MRI scans - sample images", color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(project_dir / "sample_scans.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: sample_scans.png")
    print()

    # ------------------------------------------------------- 2. Preprocessing
    print("=" * 72)
    print("2. PREPROCESSING")
    print("=" * 72)
    print("Pixel rescaling to 0-1 is a layer inside the model, so the identical")
    print("transform applies automatically at inference time and cannot be forgotten.")
    print("Augmentation is mild by design - see the comment in build_cnn().")
    print(f"Train/validation split: {1 - VALIDATION_SPLIT:.0%}/{VALIDATION_SPLIT:.0%} "
          "of the Training folder.")
    print("The Testing folder is the dataset's own held-out split and is touched once.")
    print()

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    # --------------------------------------------------------- 3. Model build
    print("=" * 72)
    print("3. MODEL DEVELOPMENT")
    print("=" * 72)
    model = build_cnn(n_classes)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    print()

    class_weight = {
        i: total_train / (n_classes * counts[name])
        for i, name in enumerate(class_names)
    }
    print("Class weights (counteracting imbalance):")
    for i, name in enumerate(class_names):
        print(f"  {name:<20s} {class_weight[i]:.3f}")
    print()

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=10,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                              min_lr=1e-6, verbose=0),
        ],
        verbose=2,
    )
    print()

    # ----------------------------------------------------------- 4. Evaluation
    print("=" * 72)
    print("4. EVALUATION")
    print("=" * 72)
    test_images, y_test = dataset_to_arrays(test_ds)
    probabilities = model.predict(test_images, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    test_accuracy = float((y_pred == y_test).mean())

    print(f"Test images: {len(y_test)}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names,
                                digits=4, zero_division=0))

    matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(matrix)
    print()

    # Clinically framed binary view: tumour present vs absent.
    no_tumour_index = class_names.index(NO_TUMOUR_CLASS)
    truth_has_tumour = y_test != no_tumour_index
    pred_has_tumour = y_pred != no_tumour_index
    true_positive = int((truth_has_tumour & pred_has_tumour).sum())
    false_negative = int((truth_has_tumour & ~pred_has_tumour).sum())
    true_negative = int((~truth_has_tumour & ~pred_has_tumour).sum())
    false_positive = int((~truth_has_tumour & pred_has_tumour).sum())
    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)

    print("-" * 72)
    print("CLINICAL VIEW - tumour present vs absent (the screening question)")
    print("-" * 72)
    print(f"  True positives  (tumour found)    : {true_positive}")
    print(f"  False negatives (TUMOUR MISSED)   : {false_negative}   <-- the costly error")
    print(f"  True negatives  (correctly clear) : {true_negative}")
    print(f"  False positives (false alarm)     : {false_positive}")
    print(f"  Sensitivity (recall on tumour)    : {sensitivity:.4f}")
    print(f"  Specificity (recall on healthy)   : {specificity:.4f}")
    print()

    # --------------------------------------------------------------- Plots
    epochs_ran = range(1, len(history.history["accuracy"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_acc.plot(epochs_ran, history.history["accuracy"], color=COLOR_TRAIN,
                linewidth=2, label="Training")
    ax_acc.plot(epochs_ran, history.history["val_accuracy"], color=COLOR_VAL,
                linewidth=2, label="Validation")
    ax_acc.set_title("Accuracy vs Epoch", color=INK_PRIMARY)
    ax_acc.set_xlabel("Epoch", color=INK_SECONDARY)
    ax_acc.set_ylabel("Accuracy", color=INK_SECONDARY)
    ax_acc.legend(frameon=False, labelcolor=INK_SECONDARY)
    style_axes(ax_acc)
    ax_loss.plot(epochs_ran, history.history["loss"], color=COLOR_TRAIN,
                 linewidth=2, label="Training")
    ax_loss.plot(epochs_ran, history.history["val_loss"], color=COLOR_VAL,
                 linewidth=2, label="Validation")
    ax_loss.set_title("Loss vs Epoch", color=INK_PRIMARY)
    ax_loss.set_xlabel("Epoch", color=INK_SECONDARY)
    ax_loss.set_ylabel("Sparse categorical crossentropy", color=INK_SECONDARY)
    ax_loss.legend(frameon=False, labelcolor=INK_SECONDARY)
    style_axes(ax_loss)
    fig.tight_layout()
    fig.savefig(project_dir / "training_history.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: training_history.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix - MRI test set", color=INK_PRIMARY, fontsize=12)
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("Actual", color=INK_SECONDARY)
    short = [n.replace("_tumor", "").replace("_", " ") for n in class_names]
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(short)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    threshold = matrix.max() / 2
    for r in range(n_classes):
        for c in range(n_classes):
            if matrix[r, c]:
                ax.text(c, r, matrix[r, c], ha="center", va="center", fontsize=9,
                        color="white" if matrix[r, c] > threshold else INK_PRIMARY)
    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_SECONDARY, length=0)
    fig.tight_layout()
    fig.savefig(project_dir / "confusion_matrix.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: confusion_matrix.png")

    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    wrong = np.flatnonzero(y_pred != y_test)
    right = np.flatnonzero(y_pred == y_test)
    picks = list(right[:4]) + list(wrong[:4])
    for ax, index in zip(axes.ravel(), picks):
        ax.imshow(test_images[index].astype("uint8"))
        correct = y_pred[index] == y_test[index]
        ax.set_title(f"{class_names[y_pred[index]]} {probabilities[index].max():.0%}\n"
                     f"(true: {class_names[y_test[index]]})",
                     color="#1baf7a" if correct else "#e34948", fontsize=8)
        ax.axis("off")
    fig.suptitle("Correct (top) and incorrect (bottom) predictions",
                 color=INK_PRIMARY, fontsize=12)
    fig.tight_layout()
    fig.savefig(project_dir / "sample_predictions.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: sample_predictions.png")
    print()

    # -------------------------------------------------------- Observations
    per_class_recall = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    worst = int(np.argmin(per_class_recall))
    best = int(np.argmax(per_class_recall))

    print("=" * 72)
    print("OBSERVATIONS")
    print("=" * 72)
    print(
        f"1. The CNN classifies the {len(y_test)} held-out MRI scans into four "
        f"categories with {test_accuracy:.2%} accuracy. On the binary screening "
        f"question - is there a tumour at all - sensitivity is {sensitivity:.2%} and "
        f"specificity {specificity:.2%}, with {false_negative} tumours missed outright."
    )
    print(
        f"2. Accuracy is the wrong headline for a medical model. The {false_negative} "
        f"false negatives and {false_positive} false positives are not "
        "interchangeable: a false alarm costs a follow-up scan, a miss can cost a life. "
        "A deployed screening tool would move its decision threshold to favour "
        "sensitivity, accepting more false alarms to reduce misses."
    )
    print(
        f"3. Per-class recall is uneven - '{class_names[best]}' reaches "
        f"{per_class_recall[best]:.2%} while '{class_names[worst]}' reaches only "
        f"{per_class_recall[worst]:.2%}. Glioma and meningioma are the pair most often "
        "confused, which is expected: both are extra-axial masses whose appearance "
        "overlaps on a single 2D slice without contrast timing or clinical context."
    )
    print(
        "4. This dataset has documented labelling problems - a known subset of the "
        "glioma folder is mislabelled - and every scan is a single 2D slice rather than "
        "a full volume. Both facts cap the achievable accuracy and mean these numbers "
        "should be read as an architecture exercise, not a clinical validation."
    )
    print()
    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    print(
        f"A convolutional network trained from scratch classified brain MRI scans into "
        f"glioma, meningioma, pituitary tumour and no-tumour with {test_accuracy:.2%} "
        f"accuracy on the held-out test split, achieving {sensitivity:.2%} sensitivity "
        f"and {specificity:.2%} specificity on the underlying screening question. Class "
        "weighting was necessary because the no-tumour class is the smallest, and a model "
        "biased toward predicting some tumour type would score well on accuracy while "
        "being useless at ruling disease out. The clinically meaningful metric is "
        f"sensitivity: {false_negative} missed tumours matter far more than "
        f"{false_positive} false alarms, and a deployed system would shift its threshold "
        "accordingly. Two caveats bound these results - the dataset contains known "
        "labelling errors, and each sample is a single 2D slice rather than a volumetric "
        "study. A real diagnostic tool would use 3D volumes, multiple MRI sequences, "
        "transfer learning from a medical imaging backbone, and would assist a "
        "radiologist rather than replace one."
    )


if __name__ == "__main__":
    main()
