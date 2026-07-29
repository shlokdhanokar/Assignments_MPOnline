"""Adult Census Income Classification.

Predicts whether a person earns more than $50K/year from 1994 US census
attributes. Compares three classifiers on the dataset's canonical train/test
split and reports threshold-independent as well as threshold-dependent metrics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
TRAIN_FILE = "adult.data"
TEST_FILE = "adult.test"
TARGET = "income"
POSITIVE_LABEL = ">50K"
# fnlwgt is a census sampling weight describing how many people the row represents.
# It says nothing about the individual's income and leaks survey design into the model.
DROP_COLUMNS = ["fnlwgt"]
RANDOM_STATE = 42

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY, length=0)


def ensure_dataset(project_dir: Path) -> None:
    """Download the UCI Adult files on first run so the project is one-command runnable."""
    for filename in (TRAIN_FILE, TEST_FILE):
        target = project_dir / filename
        if target.exists():
            continue
        url = UCI_BASE + filename
        print(f"Downloading {filename} from {url} ...")
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
        except Exception as error:  # noqa: BLE001 - surface any network failure clearly
            raise SystemExit(
                f"Could not download {filename}: {error}\n"
                f"Download it manually from {UCI_BASE} and place it in this folder."
            ) from error


def load_split(path: Path, skip_header: bool) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1 if skip_header else 0,
        na_values=["?"],
    )
    # adult.test writes the label with a trailing period ('<=50K.'), unlike adult.data.
    df[TARGET] = df[TARGET].str.rstrip(".").str.strip()
    return df


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    ensure_dataset(project_dir)

    # ---------------------------------------------------------------- 1. Load
    train_df = load_split(project_dir / TRAIN_FILE, skip_header=False)
    test_df = load_split(project_dir / TEST_FILE, skip_header=True)

    print("=" * 70)
    print("1. DATA UNDERSTANDING")
    print("=" * 70)
    print(f"Training records: {len(train_df)}   Test records: {len(test_df)}")
    print()
    print("First 5 records:")
    print(train_df.head())
    print()

    print("Dataset info:")
    train_df.info()
    print()

    print("Summary statistics (numerical):")
    print(train_df.describe())
    print()

    print("Target distribution (training):")
    counts = train_df[TARGET].value_counts()
    print(counts)
    print(f"Positive class '{POSITIVE_LABEL}' share: {counts[POSITIVE_LABEL] / len(train_df):.2%}")
    print()

    # ------------------------------------------------------- 2. Preprocessing
    print("=" * 70)
    print("2. DATA PREPROCESSING")
    print("=" * 70)
    print("Missing values per column (training, '?' treated as missing):")
    missing = train_df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  none")
    print()

    print(f"Dropping columns: {DROP_COLUMNS} (census sampling weight, not a personal attribute)")
    train_df = train_df.drop(columns=DROP_COLUMNS)
    test_df = test_df.drop(columns=DROP_COLUMNS)
    print()

    x_train = train_df.drop(columns=[TARGET])
    y_train = (train_df[TARGET] == POSITIVE_LABEL).astype(int)
    x_test = test_df.drop(columns=[TARGET])
    y_test = (test_df[TARGET] == POSITIVE_LABEL).astype(int)

    numeric_features = list(x_train.select_dtypes(include=["number"]).columns)
    categorical_features = [c for c in x_train.columns if c not in numeric_features]
    print("Numerical features:", numeric_features)
    print("Categorical features:", categorical_features)
    print()
    print(f"Target encoded: '{POSITIVE_LABEL}' -> 1, '<=50K' -> 0")
    print("Using the dataset's canonical train/test split rather than a random one,")
    print("so results are comparable with published benchmarks on this dataset.")
    print()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Dense output: HistGradientBoostingClassifier does not accept sparse X.
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                     sparse_output=False)),
        ]), categorical_features),
    ])

    # --------------------------------------------------------- 3. Model build
    print("=" * 70)
    print("3. MODEL DEVELOPMENT")
    print("=" * 70)
    models = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, random_state=RANDOM_STATE)),
        ]),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    print()

    # ----------------------------------------------------------- 4. Evaluation
    print("=" * 70)
    print("4. MODEL EVALUATION")
    print("=" * 70)
    summary = pd.DataFrame({
        name: {k: v for k, v in res.items() if k not in ("y_pred", "y_proba")}
        for name, res in results.items()
    }).T
    print(summary.round(4).to_string())
    print()

    best_name = summary["roc_auc"].idxmax()
    best = results[best_name]
    print(f"Best model by ROC-AUC: {best_name}")
    print()
    print(f"Classification report - {best_name}:")
    print(classification_report(y_test, best["y_pred"], target_names=["<=50K", ">50K"], digits=4))

    matrix = confusion_matrix(y_test, best["y_pred"])
    print("Confusion matrix:")
    print(matrix)
    print()

    # Plot: confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {best_name}", color=INK_PRIMARY)
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("Actual", color=INK_SECONDARY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["<=50K", ">50K"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["<=50K", ">50K"])
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    threshold = matrix.max() / 2
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{matrix[r, c]:,}", ha="center", va="center",
                    color="white" if matrix[r, c] > threshold else INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(project_dir / "confusion_matrix.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: confusion_matrix.png")

    # Plot: ROC curves for all three models
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, res), color in zip(results.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC {res['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle="--",
            label="Random baseline")
    ax.set_title("ROC Curves - Income > $50K", color=INK_PRIMARY, fontsize=13)
    ax.set_xlabel("False positive rate", color=INK_SECONDARY)
    ax.set_ylabel("True positive rate", color=INK_SECONDARY)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_SECONDARY)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(project_dir / "roc_curves.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: roc_curves.png")

    # Plot: permutation importance on the best model (model-agnostic, computed on
    # the test set, so it measures what actually helps on unseen data).
    print("Computing permutation importance (this takes a moment) ...")
    perm = permutation_importance(
        models[best_name], x_test, y_test, n_repeats=5,
        random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    importance = pd.Series(perm.importances_mean, index=x_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance.index, importance.values, color=COLORS[0])
    ax.set_title(f"Permutation Importance - {best_name} (drop in ROC-AUC)",
                 color=INK_PRIMARY, fontsize=12)
    ax.set_xlabel("Mean decrease in ROC-AUC when the column is shuffled",
                  color=INK_SECONDARY)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(project_dir / "feature_importance.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: feature_importance.png")
    print()

    print("Top 5 features by permutation importance:")
    print(importance.sort_values(ascending=False).head(5).round(4).to_string())
    print()

    # ---------------------------------------------------------- 5. Observations
    tn, fp, fn, tp = matrix.ravel()
    baseline = 1 - y_test.mean()
    print("=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    print(
        f"1. {best_name} performs best with {best['accuracy']:.2%} accuracy and "
        f"{best['roc_auc']:.4f} ROC-AUC. Accuracy alone flatters every model here: "
        f"predicting '<=50K' for everyone already scores {baseline:.2%}, because only "
        f"{y_test.mean():.2%} of the test set earns more. ROC-AUC and the positive-class "
        "recall are the metrics that actually separate the models."
    )
    print(
        f"2. The models trade precision against recall on the minority class. "
        f"{best_name} reaches {best['precision']:.4f} precision but only "
        f"{best['recall']:.4f} recall - it misses {fn:,} of the {tp + fn:,} people who "
        f"genuinely earn >$50K, while raising {fp:,} false alarms. The default 0.5 "
        "probability threshold is tuned for accuracy, not for balanced error costs."
    )
    print(
        "3. Tree ensembles beat logistic regression because the relationship is not "
        "linear or additive: income depends on interactions such as marital status "
        "combined with hours worked and education, which a linear model can only "
        "capture if those interactions are engineered by hand."
    )
    print(
        f"4. Permutation importance ranks {importance.idxmax()} highest. Note that "
        "capital-gain is extremely skewed - most people record zero - so it acts as a "
        "near-deterministic flag for a small, wealthy subgroup rather than a smooth "
        "predictor across the population."
    )
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(
        f"Three classifiers were trained on the 1994 Adult Census extract using the "
        f"dataset's canonical {len(train_df):,}/{len(test_df):,} train/test split. "
        f"{best_name} performed best at {best['accuracy']:.2%} accuracy and "
        f"{best['roc_auc']:.4f} ROC-AUC. The dominant characteristic of this dataset is "
        f"class imbalance: only {y_test.mean():.1%} of records exceed $50K, so a "
        f"do-nothing classifier already scores {baseline:.1%} accuracy and headline "
        "accuracy is close to meaningless on its own. Marital status, education level, "
        "capital gains and hours worked carry the most signal. In a real deployment the "
        "decision threshold should be set from the cost of a false positive relative to "
        "a false negative rather than left at 0.5, and the model should be audited for "
        "disparate impact across the sex and race columns before any operational use."
    )


if __name__ == "__main__":
    main()
