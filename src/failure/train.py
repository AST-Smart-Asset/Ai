"""
Train the 90-day asset-failure classifier (Logistic Regression pipeline).

Usage:
    python -m scripts.train_failure_model \
        --assets data/tables/assets.csv \
        --events data/tables/asset_events.csv \
        --work-orders data/tables/work_orders.csv
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FAILURE_FEATURES,
    FAILURE_FEATURES_PATH,
    FAILURE_HORIZON_DAYS,
    FAILURE_METADATA_PATH,
    FAILURE_MODEL_PATH,
    FAILURE_THRESHOLD,
)
from src.failure.feature_engineering import (
    build_failure_events,
    build_labeled_dataset,
    load_tables,
    mark_corrective_orders,
    valid_window,
)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            ),
        ]
    )


def time_based_split(df, split_ratio: float = 0.8):
    """Chronological (not random) split — avoids leaking future data into training."""
    split_index = int(len(df) * split_ratio)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def evaluate(y_true, y_pred) -> dict:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
    }


def due_date_baseline(df, horizon: int) -> dict:
    """Deterministic baseline: flag 'failure' whenever maintenance is due within `horizon` days."""
    baseline_pred = df["days_to_maintenance"].between(0, horizon).astype(int)
    return evaluate(df[f"failure_{horizon}d"], baseline_pred)


def run(assets_path, events_path, work_orders_path, horizon=FAILURE_HORIZON_DAYS, output_dir=None):
    model_path = FAILURE_MODEL_PATH if output_dir is None else output_dir / "failure_classifier_90d.pkl"
    features_path = (
        FAILURE_FEATURES_PATH if output_dir is None else output_dir / "failure_classifier_features.json"
    )
    metadata_path = (
        FAILURE_METADATA_PATH if output_dir is None else output_dir / "failure_classifier_90d_metadata.json"
    )

    assets, _events, work_orders = load_tables(assets_path, events_path, work_orders_path)
    model_data = build_labeled_dataset(assets, work_orders, horizons=(30, 60, horizon))

    wo = mark_corrective_orders(work_orders)
    failure_events = build_failure_events(wo)
    valid_data = valid_window(model_data, failure_events, horizon)

    train_df, test_df = time_based_split(valid_data)
    X_train, y_train = train_df[FAILURE_FEATURES], train_df[f"failure_{horizon}d"].astype(int)
    X_test, y_test = test_df[FAILURE_FEATURES], test_df[f"failure_{horizon}d"].astype(int)

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= FAILURE_THRESHOLD).astype(int)

    metrics = evaluate(y_test, y_pred)
    baseline_metrics = due_date_baseline(test_df, horizon)

    joblib.dump(model, model_path)
    with open(features_path, "w") as f:
        json.dump(FAILURE_FEATURES, f, indent=2)

    metadata = {
        "model_name": f"{horizon}-day Asset Failure Classifier",
        "model_type": "Logistic Regression",
        "target": f"failure_within_{horizon}_days",
        "threshold": FAILURE_THRESHOLD,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "test_positive_cases": int(y_test.sum()),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "failure_label_definition": f"Future corrective/fault-related work order within {horizon} days",
        "label_proxy": "completion_notes contains 'corrective job' or 'fault report'",
        "status": "experimental",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Small number of observed failure cases",
            "Failure labels are a proxy rather than confirmed failure labels",
            "Synthetic/limited project data limits generalization",
            "Model must not automatically change asset or work-order state",
        ],
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"=== {horizon}-Day Logistic Regression ===")
    print(metrics)
    print("\n=== Due-date deterministic baseline ===")
    print(baseline_metrics)
    print(f"\nSaved model to    {model_path}")
    print(f"Saved features to {features_path}")
    print(f"Saved metadata to {metadata_path}")

    return {"metrics": metrics, "baseline": baseline_metrics, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description="Train the N-day failure classifier.")
    parser.add_argument("--assets", required=True, help="Path to assets.csv")
    parser.add_argument("--events", required=True, help="Path to asset_events.csv")
    parser.add_argument("--work-orders", required=True, help="Path to work_orders.csv")
    parser.add_argument("--horizon", type=int, default=FAILURE_HORIZON_DAYS)
    args = parser.parse_args()
    run(args.assets, args.events, args.work_orders, horizon=args.horizon)


if __name__ == "__main__":
    main()
