"""Train and serialise the income-prediction model served by app.py.

Run this once before starting the web app (or let build.sh run it during the
Render build). The trained pipeline is written to model.joblib together with the
metadata the web form needs to render its dropdowns.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
DATA_FILE = "adult.data"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
# Kept deliberately small: a public web form with 14 fields is unusable, and these
# carry most of the signal (see project 01's permutation-importance results).
FEATURES = [
    "age", "education-num", "hours-per-week", "capital-gain", "capital-loss",
    "workclass", "marital-status", "occupation", "relationship", "sex",
]
TARGET = "income"
POSITIVE_LABEL = ">50K"
RANDOM_STATE = 42

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
METADATA_PATH = Path(__file__).resolve().parent / "model_metadata.json"


def ensure_dataset(project_dir: Path) -> Path:
    target = project_dir / DATA_FILE
    if not target.exists():
        print(f"Downloading {DATA_FILE} ...")
        with urllib.request.urlopen(UCI_BASE + DATA_FILE, timeout=120) as response:
            target.write_bytes(response.read())
    return target


def train() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = ensure_dataset(project_dir)

    df = pd.read_csv(data_path, names=COLUMNS, skipinitialspace=True, na_values=["?"])
    df[TARGET] = df[TARGET].str.rstrip(".").str.strip()

    x = df[FEATURES]
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)

    numeric = list(x.select_dtypes(include=["number"]).columns)
    categorical = [c for c in FEATURES if c not in numeric]

    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ])),
        ("clf", HistGradientBoostingClassifier(max_iter=200, random_state=RANDOM_STATE)),
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    print(f"Holdout accuracy: {accuracy:.4f}   ROC-AUC: {auc:.4f}")

    joblib.dump(model, MODEL_PATH)
    metadata = {
        "features": FEATURES,
        "numeric": numeric,
        "categorical": {c: sorted(df[c].dropna().unique().tolist()) for c in categorical},
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(auc), 4),
        "training_rows": int(len(x_train)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) "
          f"and {METADATA_PATH.name}")


if __name__ == "__main__":
    train()
