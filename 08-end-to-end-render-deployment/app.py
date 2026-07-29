"""Flask web service exposing the income-prediction model.

Routes
    GET  /            HTML form for interactive predictions
    POST /            same form, rendered with the prediction result
    POST /api/predict JSON API -> {"prediction": ..., "probability": ...}
    GET  /health      liveness probe for the platform health check
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

app = Flask(__name__)

# Loaded once at import time, not per request: deserialising the pipeline on every
# call would dominate response latency and defeat the point of a served model.
if not MODEL_PATH.exists():
    raise SystemExit(
        f"{MODEL_PATH.name} not found. Run 'python train_model.py' first "
        "(the Render build command does this automatically)."
    )
MODEL = joblib.load(MODEL_PATH)
METADATA = json.loads(METADATA_PATH.read_text())

NUMERIC_FIELDS = METADATA["numeric"]
CATEGORICAL_FIELDS = METADATA["categorical"]
FEATURES = METADATA["features"]


def build_frame(payload: dict) -> pd.DataFrame:
    """Coerce an incoming payload into the single-row frame the pipeline expects."""
    row = {}
    missing = []
    for field in FEATURES:
        if field not in payload or payload[field] in ("", None):
            missing.append(field)
            continue
        value = payload[field]
        if field in NUMERIC_FIELDS:
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"'{field}' must be a number, got {payload[field]!r}") from error
        row[field] = value

    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    return pd.DataFrame([row], columns=FEATURES)


def predict(payload: dict) -> dict:
    frame = build_frame(payload)
    probability = float(MODEL.predict_proba(frame)[0, 1])
    return {
        "prediction": ">50K" if probability >= 0.5 else "<=50K",
        "probability_above_50k": round(probability, 4),
        "confidence": round(max(probability, 1 - probability), 4),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    submitted = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            result = predict(submitted)
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        numeric_fields=NUMERIC_FIELDS,
        categorical_fields=CATEGORICAL_FIELDS,
        submitted=submitted,
        result=result,
        error=error,
        metadata=METADATA,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON (set Content-Type: application/json)."}), 400
    try:
        return jsonify(predict(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "accuracy": METADATA["accuracy"],
        "roc_auc": METADATA["roc_auc"],
    })


if __name__ == "__main__":
    # Render supplies the port to bind through $PORT.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
