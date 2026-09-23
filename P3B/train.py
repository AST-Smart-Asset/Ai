import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from lightgbm import LGBMClassifier

data = pd.read_csv("cleaned_ml_training_snapshots.csv")

leakage_cols = [
    "asset_tag",
    "snapshot_date",
    "days_to_next_failure",
    "missing_days_to_failure",
    "failure_status",
    "age_years",
    "age_group",
    "cost_group",
    "snapshot_month_name"
]
df = data.drop(columns=[c for c in leakage_cols if c in data.columns])

target = "failure_within_30d"
X = df.drop(columns=[target])
y = df[target]

cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

clf = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=20,
    min_child_samples=20,
    scale_pos_weight=5.0,
    random_state=42,
    verbosity=-1
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

y_pred_tuned = (y_prob >= best_threshold).astype(int)

print(f"Optimal Threshold: {best_threshold:.4f}\n")
print(classification_report(y_test, y_pred_tuned))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR AUC: {average_precision_score(y_test, y_prob):.4f}\n")

cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
all_features = num_cols + encoded_cat_names
importances = model.named_steps["classifier"].feature_importances_

feature_ranking = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

print("Top 10 Influential Features:")
print(feature_ranking.head(10).to_string(index=False))

artifacts = {
    "model": model,
    "optimal_threshold": best_threshold
}
joblib.dump(artifacts, "model_artifacts.pkl")
print("\nModel saved successfully as 'model_artifacts.pkl'")