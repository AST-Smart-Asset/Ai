# Run this in the Colab notebook where the trained model files already exist.
from pathlib import Path
import shutil

src = Path("/content")
dst = Path("/content/predictive_maintenance_ai")
dst.mkdir(exist_ok=True)

required = [
    "rul_random_forest.pkl",
    "features.json",
    "failure_classifier_90d.pkl",
    "failure_classifier_features.json",
]

for name in required:
    source = src / name
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {name}. Save/download the trained model file into /content first."
        )
    shutil.copy2(source, dst / name)

print("Copied:")
for p in dst.iterdir():
    print(" -", p.name)

# Optional: download the complete deployment folder as a ZIP.
import shutil
zip_path = shutil.make_archive(
    "/content/predictive_maintenance_ai",
    "zip",
    root_dir=dst.parent,
    base_dir=dst.name,
)
print("ZIP:", zip_path)
