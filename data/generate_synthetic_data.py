"""
SAHAYAK — Synthetic CRPF-Style HRMS Data Generator
===================================================
Generates realistic synthetic personnel data with correlated stress indicators
for the AI-Based Predictive Personnel Stress & Welfare Monitoring System.

This is NOT real CRPF data. It is entirely fictional and created for MVP
prototype validation only.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_PERSONNEL = 2000

# ---------- Helper distributions ----------

RANKS = [
    "Constable", "Head Constable", "ASI", "SI",
    "Inspector", "Dy. Commandant", "Commandant", "DIG", "IG"
]
RANK_WEIGHTS = [0.30, 0.22, 0.15, 0.12, 0.10, 0.05, 0.03, 0.02, 0.01]

DEPLOYMENT_ZONES = [
    "J&K (High Risk)", "Northeast (High Risk)", "Naxal Belt (High Risk)",
    "Border Area (Moderate)", "Urban Posting (Low)", "Training Center (Low)",
    "HQ / Admin (Low)"
]
ZONE_RISK = {
    "J&K (High Risk)": 0.85, "Northeast (High Risk)": 0.80,
    "Naxal Belt (High Risk)": 0.82, "Border Area (Moderate)": 0.55,
    "Urban Posting (Low)": 0.25, "Training Center (Low)": 0.20,
    "HQ / Admin (Low)": 0.15
}
ZONE_WEIGHTS = [0.18, 0.12, 0.15, 0.20, 0.15, 0.10, 0.10]

UNITS = [
    "CRPF", "BSF", "CISF", "ITBP", "SSB", "NSG", "AR", "RPF"
]

MARITAL_STATUS = ["Single", "Married", "Widowed", "Divorced"]
MARITAL_WEIGHTS = [0.20, 0.68, 0.04, 0.08]


def generate_data(n: int = NUM_PERSONNEL) -> pd.DataFrame:
    """Generate *n* rows of synthetic CRPF-style HRMS + wellness data."""

    data = {}

    # --- Demographics ---
    data["personnel_id"] = [f"SAH-{i:05d}" for i in range(1, n + 1)]
    data["age"] = np.random.randint(21, 58, n)
    data["gender"] = np.random.choice(["Male", "Female"], n, p=[0.88, 0.12])
    data["rank"] = np.random.choice(RANKS, n, p=RANK_WEIGHTS)
    data["unit"] = np.random.choice(UNITS, n)
    data["years_of_service"] = np.clip(data["age"] - np.random.randint(18, 25, n), 1, 38)
    data["marital_status"] = np.random.choice(MARITAL_STATUS, n, p=MARITAL_WEIGHTS)
    data["num_dependents"] = np.where(
        np.array(data["marital_status"]) == "Single",
        np.random.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1]),
        np.random.randint(1, 6, n)
    )

    # --- Deployment / Duty ---
    data["deployment_zone"] = np.random.choice(DEPLOYMENT_ZONES, n, p=ZONE_WEIGHTS)
    data["deployment_zone_risk"] = [ZONE_RISK[z] for z in data["deployment_zone"]]
    data["deployment_duration_months"] = np.random.randint(1, 36, n)
    data["total_deployments"] = np.random.randint(1, 15, n)
    data["transfer_frequency_5yr"] = np.random.randint(0, 8, n)
    data["family_separation_months"] = np.clip(
        data["deployment_duration_months"] + np.random.randint(-3, 6, n), 0, 48
    )
    data["distance_from_home_km"] = np.random.randint(50, 3500, n)

    # --- Workload ---
    data["avg_duty_hours_per_week"] = np.round(
        np.clip(np.random.normal(52, 12, n), 35, 90), 1
    )
    data["night_duty_freq_per_month"] = np.random.randint(0, 20, n)
    data["training_hours_per_month"] = np.round(
        np.clip(np.random.normal(15, 8, n), 0, 60), 1
    )
    data["workload_score"] = np.round(
        np.clip(np.random.normal(5.5, 2.0, n), 1, 10), 1
    )

    # --- Leave ---
    data["leaves_entitled"] = np.random.choice([30, 45, 60], n, p=[0.5, 0.35, 0.15])
    data["leaves_taken"] = np.clip(
        (data["leaves_entitled"] * np.random.uniform(0.1, 1.1, n)).astype(int),
        0, data["leaves_entitled"]
    )
    data["leave_utilization_ratio"] = np.round(
        data["leaves_taken"] / data["leaves_entitled"], 3
    )
    data["sick_leave_count"] = np.random.randint(0, 15, n)

    # --- Career ---
    data["years_since_last_promotion"] = np.random.randint(0, 12, n)
    data["commendations"] = np.random.randint(0, 8, n)
    data["disciplinary_actions"] = np.random.choice(
        [0, 1, 2, 3], n, p=[0.70, 0.18, 0.08, 0.04]
    )
    data["incident_exposure_count"] = np.random.randint(0, 20, n)

    # --- Self-reported wellness (optional / voluntary) ---
    data["wellness_checkin_score"] = np.round(
        np.clip(np.random.normal(6.0, 2.0, n), 1, 10), 1
    )
    data["sleep_quality_score"] = np.round(
        np.clip(np.random.normal(3.0, 1.0, n), 1, 5), 1
    )
    data["physical_fitness_score"] = np.round(
        np.clip(np.random.normal(6.5, 2.0, n), 1, 10), 1
    )
    data["mood_score"] = np.round(
        np.clip(np.random.normal(5.5, 2.0, n), 1, 10), 1
    )
    data["social_connectedness_score"] = np.round(
        np.clip(np.random.normal(5.0, 2.0, n), 1, 10), 1
    )

    df = pd.DataFrame(data)

    # ---- Generate correlated burnout risk score (target) ----
    # Weighted combination of features that logically contribute to stress
    risk = (
        0.15 * df["deployment_zone_risk"]
        + 0.08 * (df["deployment_duration_months"] / 36)
        + 0.07 * (df["avg_duty_hours_per_week"] / 90)
        + 0.06 * (df["night_duty_freq_per_month"] / 20)
        + 0.06 * (df["family_separation_months"] / 48)
        + 0.05 * (df["workload_score"] / 10)
        + 0.05 * (1 - df["leave_utilization_ratio"])
        + 0.05 * (df["sick_leave_count"] / 15)
        + 0.04 * (df["years_since_last_promotion"] / 12)
        + 0.04 * (df["incident_exposure_count"] / 20)
        + 0.04 * (df["disciplinary_actions"] / 3)
        + 0.03 * (df["transfer_frequency_5yr"] / 8)
        + 0.03 * (df["distance_from_home_km"] / 3500)
        - 0.06 * (df["wellness_checkin_score"] / 10)
        - 0.05 * (df["sleep_quality_score"] / 5)
        - 0.04 * (df["physical_fitness_score"] / 10)
        - 0.04 * (df["mood_score"] / 10)
        - 0.03 * (df["social_connectedness_score"] / 10)
        - 0.03 * (df["commendations"] / 8)
    )

    # Add controlled noise
    noise = np.random.normal(0, 0.06, n)
    risk = np.clip(risk + noise + 0.20, 0.0, 1.0)  # shift baseline up

    df["burnout_risk_score"] = np.round(risk, 3)

    # Binary risk label
    df["stress_label"] = (df["burnout_risk_score"] >= 0.5).astype(int)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_data()
    df.to_csv("data/synthetic_crpf_data.csv", index=False)
    print(f"✅  Generated {len(df)} records → data/synthetic_crpf_data.csv")
    print(f"   Stress positive rate: {df['stress_label'].mean():.1%}")
    print(f"   Burnout risk — mean: {df['burnout_risk_score'].mean():.3f}  "
          f"std: {df['burnout_risk_score'].std():.3f}")
    print(df.head())
