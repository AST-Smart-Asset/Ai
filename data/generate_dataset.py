import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

def generate_synthetic_telemetry(n_samples: int = 12000, output_path: str = "data/asset_telemetry.csv"):
    categories = [
        ("HVAC", 1.8, 0.6, 52.0, 7.0, 22.0),
        ("Generator", 2.2, 0.8, 68.0, 8.5, 45.0),
        ("Medical", 1.1, 0.3, 38.0, 4.0, 8.0),
        ("Lab Equipment", 1.3, 0.4, 42.0, 5.0, 12.0),
        ("IT Server", 0.9, 0.2, 45.0, 6.0, 15.0),
        ("Elevator", 2.0, 0.7, 55.0, 6.5, 28.0),
        ("Water Pump", 2.4, 0.9, 58.0, 7.5, 32.0),
    ]

    records = []

    for i in range(n_samples):
        cat_name, base_vib, vib_std, base_temp, temp_std, base_pwr = random.choice(categories)
        asset_id = f"AST-{cat_name[:3].upper()}-{random.randint(100, 999)}"

        age_days = random.randint(30, 2500)
        days_since_pm = random.randint(1, 240)
        hist_wo_count = int(np.random.poisson(lam=max(0.5, age_days / 365.0 * 2.0)))
        cum_downtime = round(hist_wo_count * np.random.exponential(scale=6.0), 1)

        # Baseline noise
        vib = max(0.4, np.random.normal(base_vib, vib_std))
        temp = max(25.0, np.random.normal(base_temp, temp_std))
        power = max(2.0, np.random.normal(base_pwr, base_pwr * 0.15))
        humidity = np.clip(np.random.normal(50.0, 12.0), 20.0, 85.0)

        # Degradation effects (wear over time & overdue PM)
        pm_effect = max(0.0, (days_since_pm - 90) / 60.0)
        age_effect = (age_days / 2000.0) * 0.5
        downtime_effect = min(1.5, cum_downtime / 80.0)

        # Apply degradation to sensors
        vib += (pm_effect * 1.8) + (age_effect * 1.2) + (downtime_effect * 0.8)
        temp += (pm_effect * 12.0) + (age_effect * 8.0) + (downtime_effect * 6.0)
        power += (pm_effect * 5.0)

        # Derived features
        tv_index = (temp / 60.0) * (vib / 2.5)
        pm_ratio = float(days_since_pm) / 90.0
        dt_intensity = float(cum_downtime) / max(1.0, float(age_days) / 30.0)

        # True latent probability of failure within 30 days
        z = (
            -4.5
            + 0.55 * (vib - 2.8)
            + 0.06 * (temp - 60.0)
            + 0.015 * (days_since_pm - 90.0)
            + 0.0008 * (age_days - 700.0)
            + 0.025 * cum_downtime
            + 0.45 * (tv_index - 1.0)
        )
        true_prob = 1.0 / (1.0 + np.exp(-z))
        true_prob = np.clip(true_prob, 0.01, 0.99)

        # Binary label
        failed_30d = 1 if random.random() < true_prob else 0

        records.append({
            "asset_id": asset_id,
            "asset_type": cat_name,
            "age_days": age_days,
            "days_since_last_pm": days_since_pm,
            "cumulative_downtime_hours": cum_downtime,
            "historical_work_order_count": hist_wo_count,
            "vibration_rms": round(vib, 2),
            "operating_temperature_c": round(temp, 1),
            "power_consumption_kwh": round(power, 1),
            "ambient_humidity": round(humidity, 1),
            "thermal_vibration_index": round(tv_index, 3),
            "pm_overdue_ratio": round(pm_ratio, 3),
            "downtime_intensity": round(dt_intensity, 3),
            "failed_within_30_days": failed_30d,
        })

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records. Failure rate: {df['failed_within_30_days'].mean():.2%}. Saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_telemetry()
