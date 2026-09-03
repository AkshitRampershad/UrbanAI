"""
ML site-ranking model: predicts development_suitability (Low/Medium/High)
for a parcel from its Gold-layer features.

Trains a small set of candidate model families and keeps the
best-performing one on a held-out test set - a local, honest stand-in
for what Databricks AutoML would do at scale (searching a much larger
space of model families/hyperparameters in parallel). See
pipeline/cloud_adapters.py for where the real AutoML/MLflow calls
would plug in.

Every run is logged to models/mlflow_runs.json (run id, params, metrics,
timestamp) as a lightweight local mirror of MLflow experiment tracking.
IMPORTANT: this module reports whatever accuracy the trained model
actually achieves on the held-out test set - it is never hardcoded.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GOLD_PATH = Path("data/gold/parcels_gold.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "lot_size_acres", "distance_to_highway_mi", "distance_to_transit_mi",
    "population_density_per_sqmi", "median_income_nearby", "assessed_value",
]
BOOLEAN_FEATURES = ["utilities_available", "floodplain_flag"]
CATEGORICAL_FEATURES = ["zoning_code", "land_use_current"]
TARGET = "development_suitability"

CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def _build_pipeline(estimator) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("bool", "passthrough", BOOLEAN_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def train_and_select_best(gold_path: Path = GOLD_PATH, test_size: float = 0.2, seed: int = 42) -> dict:
    df = pd.read_parquet(gold_path)
    df = df.dropna(subset=NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES + [TARGET])

    X = df[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    results = {}
    best_name, best_pipeline, best_acc = None, None, -1.0

    for name, estimator in CANDIDATE_MODELS.items():
        t0 = time.time()
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {
            "test_accuracy": round(acc, 4),
            "train_seconds": round(time.time() - t0, 2),
        }
        if acc > best_acc:
            best_name, best_pipeline, best_acc = name, pipeline, acc

    report = classification_report(y_test, best_pipeline.predict(X_test), output_dict=True)

    run_id = uuid.uuid4().hex[:8]
    run_record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates_evaluated": results,
        "selected_model": best_name,
        "test_accuracy": round(best_acc, 4),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "classification_report": report,
    }

    joblib.dump(best_pipeline, MODEL_DIR / "site_ranking_model.joblib")
    with open(MODEL_DIR / "latest_metrics.json", "w") as f:
        json.dump(run_record, f, indent=2)

    runs_log_path = MODEL_DIR / "mlflow_runs.json"
    runs_log = json.loads(runs_log_path.read_text()) if runs_log_path.exists() else []
    runs_log.append(run_record)
    runs_log_path.write_text(json.dumps(runs_log, indent=2))

    return run_record


if __name__ == "__main__":
    record = train_and_select_best()
    print(f"Selected model: {record['selected_model']} - test accuracy: {record['test_accuracy']:.2%}")
