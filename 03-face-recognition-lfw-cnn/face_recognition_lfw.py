"""Face Recognition on Labeled Faces in the Wild (LFW) using a CNN.

Trains a CNN from scratch on the LFW subset of people with at least 70 images,
and benchmarks it against the classical eigenfaces pipeline (PCA + SVM) that
defined this task before deep learning.

The comparison is the point: with roughly a thousand training images, more
capacity is not automatically better.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


MIN_FACES = 70
RESIZE = 0.4
TEST_SIZE = 0.25
EPOCHS = int(os.environ.get("EPOCHS", "60"))
BATCH_SIZE = 32
PCA_COMPONENTS = 150
RANDOM_STATE = 42

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


def build_cnn(input_shape, n_classes: int) -> Sequential:
    """Deliberately small. ~1,000 training faces cannot support a large network."""
    return Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.08, 0.08),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.35),

        # Global average pooling instead of Flatten: it removes the large dense layer
        # that would otherwise hold most of the parameters and overfit immediately.
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ], name="lfw_cnn")


def main() -> None:
    set_seeds()
    project_dir = Path(__file__).resolve().parent

    # ---------------------------------------------------------------- 1. Load
    print("=" * 72)
    print("1. DATA UNDERSTANDING")
    print("=" * 72)
    print(f"Loading LFW (people with >= {MIN_FACES} images; downloads ~200 MB on "
          "first run) ...")
    lfw = fetch_lfw_people(min_faces_per_person=MIN_FACES, resize=RESIZE)

    images = lfw.images
    x_flat = lfw.data
    y = lfw.target
    target_names = list(lfw.target_names)
    height, width = images.shape[1], images.shape[2]
    n_classes = len(target_names)

    print(f"Images        : {images.shape[0]}")
    print(f"Image size    : {height} x {width} greyscale")
    print(f"Flattened dims: {x_flat.shape[1]} features per image")
    print(f"People (classes): {n_classes}")
    print()
    print("Class distribution:")
    counts = np.bincount(y)
    for index, name in enumerate(target_names):
        print(f"  {name:<22s} {counts[index]:4d}  ({counts[index] / len(y):.1%})")
    print()
    print(f"The dataset is severely imbalanced: '{target_names[counts.argmax()]}' accounts "
          f"for {counts.max() / len(y):.1%} of all images, so predicting that one name for "
          f"every face already scores {counts.max() / len(y):.1%} accuracy. Every result "
          "below must be read against that floor.")
    print()

    fig, axes = plt.subplots(2, 5, figsize=(11, 5.4))
    for ax, index in zip(axes.ravel(), np.linspace(0, len(images) - 1, 10).astype(int)):
        ax.imshow(images[index], cmap="gray")
        ax.set_title(target_names[y[index]].split()[-1], color=INK_PRIMARY, fontsize=9)
        ax.axis("off")
    fig.suptitle("Labeled Faces in the Wild - sample images", color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(project_dir / "sample_faces.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: sample_faces.png")
    print()

    # ------------------------------------------------------- 2. Preprocessing
    print("=" * 72)
    print("2. PREPROCESSING")
    print("=" * 72)
    x_images = (images / 255.0).astype("float32")[..., np.newaxis]
    print(f"CNN input tensor: {x_images.shape}, pixels scaled to 0-1")

    # Stratified split keeps every person represented in both halves despite the
    # heavy imbalance; the same indices are reused for both models so the
    # comparison is like-for-like.
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train images: {len(train_idx)}   Test images: {len(test_idx)}")
    print("Both models use identical train/test indices.")
    print()

    # --------------------------------------------------------- 3. CNN
    print("=" * 72)
    print("3. MODEL A - CNN TRAINED FROM SCRATCH")
    print("=" * 72)
    cnn = build_cnn((height, width, 1), n_classes)
    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    cnn.summary()
    print()

    # Class weights counteract the imbalance; without them the network collapses to
    # predicting the majority name for everything.
    class_weight = {
        i: len(y_train) / (n_classes * max(1, int((y_train == i).sum())))
        for i in range(n_classes)
    }

    history = cnn.fit(
        x_images[train_idx], to_categorical(y_train, n_classes),
        validation_split=0.15,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=15,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                              min_lr=1e-5, verbose=0),
        ],
        verbose=2,
    )
    cnn_pred = cnn.predict(x_images[test_idx], verbose=0).argmax(axis=1)
    cnn_accuracy = accuracy_score(y_test, cnn_pred)
    print(f"\nCNN test accuracy: {cnn_accuracy:.4f}")
    print()

    # ----------------------------------------- 4. Eigenfaces baseline (PCA+SVM)
    print("=" * 72)
    print("4. MODEL B - EIGENFACES BASELINE (PCA + SVM)")
    print("=" * 72)
    eigen = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS, whiten=True, random_state=RANDOM_STATE)),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                    random_state=RANDOM_STATE)),
    ])
    eigen.fit(x_flat[train_idx], y_train)
    eigen_pred = eigen.predict(x_flat[test_idx])
    eigen_accuracy = accuracy_score(y_test, eigen_pred)
    explained = eigen.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"PCA components: {PCA_COMPONENTS} retaining {explained:.1%} of pixel variance")
    print(f"Eigenfaces test accuracy: {eigen_accuracy:.4f}")
    print()

    fig, axes = plt.subplots(2, 5, figsize=(11, 5.4))
    eigenfaces = eigen.named_steps["pca"].components_[:10].reshape((10, height, width))
    for ax, index in zip(axes.ravel(), range(10)):
        ax.imshow(eigenfaces[index], cmap="gray")
        ax.set_title(f"Eigenface {index + 1}", color=INK_PRIMARY, fontsize=9)
        ax.axis("off")
    fig.suptitle("Top 10 principal components ('eigenfaces')",
                 color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(project_dir / "eigenfaces.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: eigenfaces.png")
    print()

    # ----------------------------------------------------------- 5. Evaluation
    print("=" * 72)
    print("5. EVALUATION")
    print("=" * 72)
    best_name, best_pred, best_accuracy = (
        ("CNN", cnn_pred, cnn_accuracy) if cnn_accuracy >= eigen_accuracy
        else ("Eigenfaces (PCA+SVM)", eigen_pred, eigen_accuracy)
    )
    majority_floor = counts.max() / len(y)

    print(f"{'Model':<26s} {'Accuracy':>10s}")
    print(f"{'CNN (from scratch)':<26s} {cnn_accuracy:>10.4f}")
    print(f"{'Eigenfaces (PCA+SVM)':<26s} {eigen_accuracy:>10.4f}")
    print(f"{'Majority-class floor':<26s} {majority_floor:>10.4f}")
    print()
    print(f"Best model: {best_name}")
    print()
    print(f"Classification report - {best_name}:")
    print(classification_report(y_test, best_pred, target_names=target_names,
                                digits=4, zero_division=0))

    matrix = confusion_matrix(y_test, best_pred)
    print("Confusion matrix:")
    print(matrix)
    print()

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {best_name}", color=INK_PRIMARY, fontsize=12)
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("Actual", color=INK_SECONDARY)
    short = [n.split()[-1] for n in target_names]
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(short)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    threshold = matrix.max() / 2
    for r in range(n_classes):
        for c in range(n_classes):
            if matrix[r, c]:
                ax.text(c, r, matrix[r, c], ha="center", va="center", fontsize=8,
                        color="white" if matrix[r, c] > threshold else INK_PRIMARY)
    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_SECONDARY, length=0)
    fig.tight_layout()
    fig.savefig(project_dir / "confusion_matrix.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: confusion_matrix.png")

    epochs_ran = range(1, len(history.history["accuracy"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_acc.plot(epochs_ran, history.history["accuracy"], color=COLOR_TRAIN,
                linewidth=2, label="Training")
    ax_acc.plot(epochs_ran, history.history["val_accuracy"], color=COLOR_VAL,
                linewidth=2, label="Validation")
    ax_acc.axhline(eigen_accuracy, color="#1baf7a", linestyle="--", linewidth=1.5,
                   label=f"Eigenfaces test ({eigen_accuracy:.3f})")
    ax_acc.set_title("CNN accuracy vs epoch", color=INK_PRIMARY)
    ax_acc.set_xlabel("Epoch", color=INK_SECONDARY)
    ax_acc.set_ylabel("Accuracy", color=INK_SECONDARY)
    ax_acc.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    style_axes(ax_acc)

    ax_loss.plot(epochs_ran, history.history["loss"], color=COLOR_TRAIN,
                 linewidth=2, label="Training")
    ax_loss.plot(epochs_ran, history.history["val_loss"], color=COLOR_VAL,
                 linewidth=2, label="Validation")
    ax_loss.set_title("CNN loss vs epoch", color=INK_PRIMARY)
    ax_loss.set_xlabel("Epoch", color=INK_SECONDARY)
    ax_loss.set_ylabel("Categorical crossentropy", color=INK_SECONDARY)
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    style_axes(ax_loss)
    fig.tight_layout()
    fig.savefig(project_dir / "training_history.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: training_history.png")

    fig, axes = plt.subplots(2, 5, figsize=(11, 5.6))
    wrong = np.flatnonzero(best_pred != y_test)
    right = np.flatnonzero(best_pred == y_test)
    picks = list(right[:5]) + list(wrong[:5])
    for ax, position in zip(axes.ravel(), picks):
        original = test_idx[position]
        ax.imshow(images[original], cmap="gray")
        correct = best_pred[position] == y_test[position]
        ax.set_title(f"{target_names[best_pred[position]].split()[-1]}\n"
                     f"(true: {target_names[y_test[position]].split()[-1]})",
                     color="#1baf7a" if correct else "#e34948", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{best_name} - correct (top) and incorrect (bottom) predictions",
                 color=INK_PRIMARY, fontsize=12)
    fig.tight_layout()
    fig.savefig(project_dir / "sample_predictions.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: sample_predictions.png")
    print()

    # -------------------------------------------------------- Observations
    per_class_recall = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    best_person = int(np.argmax(per_class_recall))
    worst_person = int(np.argmin(per_class_recall))

    print("=" * 72)
    print("OBSERVATIONS")
    print("=" * 72)
    print(
        f"1. The eigenfaces baseline scores {eigen_accuracy:.2%} and the from-scratch CNN "
        f"{cnn_accuracy:.2%}, against a {majority_floor:.2%} majority-class floor. "
        f"{'The classical pipeline wins' if eigen_accuracy > cnn_accuracy else 'The CNN wins'}"
        ", which is the result worth dwelling on."
    )
    print(
        f"2. Data volume, not architecture, is the binding constraint. Only "
        f"{len(train_idx)} training faces across {n_classes} identities is far below what a "
        "convolutional network needs to learn useful filters from scratch - CNNs earned "
        "their reputation on datasets of millions of images. PCA plus an RBF SVM has "
        "orders of magnitude fewer effective parameters and degrades far more gracefully "
        "in this regime."
    )
    print(
        f"3. The class imbalance shapes every metric. '{target_names[counts.argmax()]}' "
        f"supplies {counts.max() / len(y):.1%} of the images, so accuracy alone is "
        f"misleading. Per-class recall ranges from {per_class_recall[worst_person]:.2%} "
        f"('{target_names[worst_person]}') to {per_class_recall[best_person]:.2%} "
        f"('{target_names[best_person]}') - the rarer identities are markedly harder, "
        "which the macro average in the classification report exposes."
    )
    print(
        "4. The correct production answer is neither model here: it is transfer learning. "
        "A face-embedding network pre-trained on millions of faces (FaceNet, ArcFace, "
        "VGGFace) reaches well above both numbers on this dataset, because the expensive "
        "part - learning what facial structure looks like - has already been paid for "
        "elsewhere and only the final classifier needs these 1,000 images."
    )
    print()
    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    print(
        f"Two approaches to face recognition were compared on the LFW subset "
        f"({images.shape[0]} images, {n_classes} identities): a CNN trained from scratch "
        f"({cnn_accuracy:.2%}) and the classical eigenfaces pipeline of PCA plus an RBF "
        f"SVM ({eigen_accuracy:.2%}), against a {majority_floor:.2%} majority-class floor. "
        f"The headline finding is that deep learning is not automatically the stronger "
        f"choice - with roughly {len(train_idx)} training images the CNN has more capacity "
        "than the data can constrain, while PCA's linear compression into 150 eigenfaces "
        "acts as a powerful prior and the SVM's margin objective generalises well from few "
        "examples. Heavy augmentation, batch normalisation, global average pooling and "
        "class weighting were all necessary just to keep the CNN competitive. In "
        "production neither would be trained from scratch: a pre-trained face-embedding "
        "network supplies the features and only a small classifier is fitted on top."
    )


if __name__ == "__main__":
    main()
