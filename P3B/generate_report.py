import pandas as pd
import numpy as np
import joblib

artifacts = joblib.load("model_artifacts.pkl")
model = artifacts["model"]
threshold = artifacts["optimal_threshold"]

df = pd.read_csv("cleaned_ml_training_snapshots.csv")
latest_snapshots = df.sort_values("snapshot_date").groupby("asset_tag").last().reset_index()

probs = model.predict_proba(latest_snapshots)[:, 1]
latest_snapshots["failure_probability"] = np.round(probs, 4)
latest_snapshots["predicted_failure_30d"] = (probs >= threshold).astype(int)

latest_snapshots["risk_level"] = pd.cut(
    latest_snapshots["failure_probability"],
    bins=[-0.01, 0.15, threshold, 0.60, 1.0],
    labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
)

report_cols = [
    "asset_tag",
    "category_code",
    "department",
    "building",
    "prior_failures_count",
    "prior_work_orders_count",
    "avg_repair_hours_so_far",
    "life_used_percentage",
    "failure_probability",
    "risk_level",
    "predicted_failure_30d"
]

report_df = latest_snapshots[report_cols].sort_values(by="failure_probability", ascending=False)
report_df.to_csv("asset_maintenance_risk_report.csv", index=False)

print(f"Report generated successfully: 'asset_maintenance_risk_report.csv'")
print(f"Total Assets Processed: {len(report_df)}")
print(f"Assets Predicted for Failure: {report_df['predicted_failure_30d'].sum()}")