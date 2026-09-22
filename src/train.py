import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
import joblib

from .config import MODELS_DIR, MODEL_FILE, METRICS_FILE, MODEL_VERSION
from .feature_engineering import FEATURE_COLUMNS
from data.generate_dataset import generate_synthetic_telemetry

def train_and_evaluate():
    data_path = Path("data/asset_telemetry.csv")
    if not data_path.exists():
        print("Telemetry dataset not found. Generating synthetic dataset...")
        df = generate_synthetic_telemetry(n_samples=12000, output_path=str(data_path))
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Target distribution: {df['failed_within_30_days'].value_counts(normalize=True).to_dict()}")

    X = df[FEATURE_COLUMNS]
    y = df["failed_within_30_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        min_samples_split=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.50).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_test, y_pred_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature importances
    importances = {
        col: round(float(imp), 4)
        for col, imp in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    metrics = {
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "test_metrics": {
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "brier_score_loss": round(brier, 4),
        },
        "confusion_matrix": {
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
        "feature_importances": importances,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE}")

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {METRICS_FILE}")

    print("\n" + "=" * 50)
    print("MODEL TRAINING & EVALUATION REPORT")
    print("=" * 50)
    print(f"ROC-AUC:          {roc_auc:.4f} (Release Gate Threshold: >= 0.85)")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"Brier Score:      {brier:.4f}")
    print(f"Top 3 Features:   {list(importances.items())[:3]}")
    print("=" * 50)

    return metrics

if __name__ == "__main__":
    train_and_evaluate()
