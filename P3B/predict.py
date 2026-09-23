import joblib
import pandas as pd
import numpy as np

artifacts = joblib.load("model_artifacts.pkl")
model = artifacts["model"]
threshold = artifacts["optimal_threshold"]

df_all = pd.read_csv("cleaned_ml_training_snapshots.csv")

sample_data = pd.concat([
    df_all.query("failure_within_30d == 1").sample(3, random_state=42),
    df_all.query("failure_within_30d == 0").sample(3, random_state=42)
]).reset_index(drop=True)

probabilities = model.predict_proba(sample_data)[:, 1]
predictions = (probabilities >= threshold).astype(int)

results = pd.DataFrame({
    "asset_tag": sample_data["asset_tag"].values,
    "failure_probability": np.round(probabilities, 4),
    "predicted_failure_30d": predictions,
    "actual_failure_30d": sample_data["failure_within_30d"].values
})

print(f"Decision Threshold: {threshold:.4f}\n")
print(results.to_string(index=False))