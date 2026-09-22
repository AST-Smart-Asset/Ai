"""
Train the RUL (Remaining Useful Life) Random Forest regressor.

Usage:
    python -m scripts.train_rul_model \
        --train-path data/cmapss/train_FD001.txt \
        --test-path  data/cmapss/test_FD001.txt \
        --rul-path   data/cmapss/RUL_FD001.txt
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import RUL_FEATURES, RUL_FEATURES_PATH, RUL_MODEL_PATH
from src.rul.data_loader import (
    last_cycle_per_engine,
    load_test_data,
    load_train_data,
    load_true_rul,
)


def split_engines(train_df, train_fraction: float = 0.8):
    """80/20 split by engine_id (not by row) so no engine leaks between sets."""
    engines = train_df["engine_id"].unique()
    split = int(len(engines) * train_fraction)
    train_engines, holdout_engines = engines[:split], engines[split:]
    return (
        train_df[train_df["engine_id"].isin(train_engines)],
        train_df[train_df["engine_id"].isin(holdout_engines)],
    )


def train_model(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}


def baseline_evaluate(y_train, y_true) -> dict:
    """Naive baseline: always predict the mean training RUL."""
    baseline_pred = np.full(len(y_true), y_train.mean())
    return evaluate(y_true, baseline_pred)


def feature_importance(model, features):
    import pandas as pd

    return (
        pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def run(train_path: str, test_path: str, rul_path: str, output_dir=None) -> dict:
    model_path = RUL_MODEL_PATH if output_dir is None else output_dir / "rul_random_forest.pkl"
    features_path = RUL_FEATURES_PATH if output_dir is None else output_dir / "features.json"

    train_df = load_train_data(train_path)
    test_raw = load_test_data(test_path)
    true_rul = load_true_rul(rul_path)

    # Internal train/holdout split (for a quick sanity-check score)
    train_data, holdout_data = split_engines(train_df)
    model = train_model(train_data[RUL_FEATURES], train_data["RUL"])
    holdout_pred = model.predict(holdout_data[RUL_FEATURES])
    holdout_metrics = evaluate(holdout_data["RUL"], holdout_pred)

    # Retrain on the full training set, then score against the official
    # C-MAPSS test labels (this is the number that should be reported).
    final_model = train_model(train_df[RUL_FEATURES], train_df["RUL"])
    test_raw = test_raw.copy()
    test_raw["Predicted_RUL"] = final_model.predict(test_raw[RUL_FEATURES])

    last_cycles = last_cycle_per_engine(test_raw)
    last_cycles["True_RUL"] = true_rul["RUL"]
    official_metrics = evaluate(last_cycles["True_RUL"], last_cycles["Predicted_RUL"])
    baseline_metrics = baseline_evaluate(train_df["RUL"], last_cycles["True_RUL"])

    importance_df = feature_importance(final_model, RUL_FEATURES)

    joblib.dump(final_model, model_path)
    with open(features_path, "w") as f:
        json.dump(RUL_FEATURES, f, indent=2)

    print("Holdout metrics :", holdout_metrics)
    print("Official metrics:", official_metrics)
    print("Baseline metrics:", baseline_metrics)
    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))
    print(f"\nSaved model to     {model_path}")
    print(f"Saved features to  {features_path}")

    return {
        "holdout": holdout_metrics,
        "official": official_metrics,
        "baseline": baseline_metrics,
        "feature_importance": importance_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the RUL Random Forest model.")
    parser.add_argument("--train-path", required=True, help="Path to train_FD001.txt")
    parser.add_argument("--test-path", required=True, help="Path to test_FD001.txt")
    parser.add_argument("--rul-path", required=True, help="Path to RUL_FD001.txt")
    args = parser.parse_args()
    run(args.train_path, args.test_path, args.rul_path)


if __name__ == "__main__":
    main()
