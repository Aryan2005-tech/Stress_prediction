"""
SAHAYAK — XGBoost Stress / Burnout Risk Prediction Model
=========================================================
Trains an XGBoost classifier on synthetic CRPF-style HRMS data.
Saves:
  • model/xgboost_stress_model.pkl   — trained XGBoost model
  • model/scaler.pkl                 — fitted StandardScaler
  • model/feature_names.pkl          — ordered feature name list
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score
)
from xgboost import XGBClassifier

# ── Paths ───────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "synthetic_crpf_data.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_stress_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# ── Feature sets ────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "years_of_service", "num_dependents",
    "deployment_zone_risk", "deployment_duration_months",
    "total_deployments", "transfer_frequency_5yr",
    "family_separation_months", "distance_from_home_km",
    "avg_duty_hours_per_week", "night_duty_freq_per_month",
    "training_hours_per_month", "workload_score",
    "leaves_entitled", "leaves_taken", "leave_utilization_ratio",
    "sick_leave_count", "years_since_last_promotion",
    "commendations", "disciplinary_actions",
    "incident_exposure_count",
    "wellness_checkin_score", "sleep_quality_score",
    "physical_fitness_score", "mood_score",
    "social_connectedness_score",
]

CATEGORICAL_FEATURES = [
    "gender", "rank", "unit", "deployment_zone", "marital_status"
]

TARGET = "stress_label"


def prepare_features(df: pd.DataFrame):
    """Encode categoricals and return (X, y, feature_names)."""
    df = df.copy()

    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(int)
    return X, y, feature_cols, label_encoders


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"📊  Loaded {len(df)} records from {DATA_PATH}")
    print(f"    Stress positive rate: {df[TARGET].mean():.1%}\n")

    X, y, feature_names, label_encoders = prepare_features(df)

    # ── Scale ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / test split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── XGBoost ─────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_weight=3,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── Evaluate ────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # ── Cross-validation ────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save artifacts ──────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅  Model saved  → {MODEL_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅  Scaler saved → {SCALER_PATH}")

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"✅  Features saved → {FEATURES_PATH}")

    # Save label encoders too
    le_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(label_encoders, f)
    print(f"✅  Label encoders saved → {le_path}")


if __name__ == "__main__":
    train()
