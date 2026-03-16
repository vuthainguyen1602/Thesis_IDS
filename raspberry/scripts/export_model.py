#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export Model — Convert PySpark Model to scikit-learn for Raspberry Pi.

Runs on the training machine (not on the Raspberry Pi).  Trains an
equivalent scikit-learn RandomForest using the same data and parameters
as the PySpark model, then serialises it as a pickle file for edge
deployment without a Spark runtime.

Usage::

    python scripts/export_model.py

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_ROOT = os.path.dirname(PROJECT_ROOT)

# Input paths
TRAIN_PARQUET = os.path.join(THESIS_ROOT, "data", "train_data.parquet")
TEST_PARQUET = os.path.join(THESIS_ROOT, "data", "test_data.parquet")
FEATURE_IMPORTANCE_CSV = os.path.join(THESIS_ROOT, "feature_importance.csv")

# Output paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "ids_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# SHAP Top-30 features (fallback if CSV not available)
DEFAULT_FEATURES = [
    "flow_duration", "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min", "fwd_packet_length_mean",
    "bwd_packet_length_max", "bwd_packet_length_mean", "bwd_packet_length_std",
    "flow_bytes_s", "flow_packets_s", "flow_iat_mean", "flow_iat_std",
    "flow_iat_max", "flow_iat_min", "fwd_iat_total", "fwd_iat_mean",
    "bwd_iat_total", "bwd_iat_mean", "fwd_psh_flags", "bwd_packets_s",
    "min_packet_length", "max_packet_length", "packet_length_mean",
    "packet_length_std", "packet_length_variance", "average_packet_size",
    "destination_port",
]


def load_data():
    """Load parquet data as Pandas DataFrames."""
    print("Loading training data...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    print(f"  Train: {len(train_df):,} rows")

    print("Loading test data...")
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"  Test:  {len(test_df):,} rows")

    return train_df, test_df


def get_feature_columns():
    """Get the list of features to use."""
    if os.path.exists(FEATURE_IMPORTANCE_CSV):
        importance_df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
        features = importance_df.head(30)["feature"].tolist()
        print(f"  Loaded {len(features)} features from {FEATURE_IMPORTANCE_CSV}")
    else:
        features = DEFAULT_FEATURES
        print(f"  Using default SHAP Top-30 features ({len(features)} features)")
    return features


def main():
    print("\n" + "=" * 60)
    print("  MODEL EXPORT: PySpark → scikit-learn (for Raspberry Pi)")
    print("=" * 60 + "\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    train_df, test_df = load_data()

    # 2. Get features
    feature_cols = get_feature_columns()

    # Filter to available columns
    available = [c for c in feature_cols if c in train_df.columns]
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        print(f"  [WARN] Missing columns: {missing}")
    feature_cols = available
    print(f"  Using {len(feature_cols)} features")

    # 3. Prepare data
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_df["label_binary"].values
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_test = test_df["label_binary"].values

    # 4. Fit StandardScaler
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train scikit-learn Random Forest (matching PySpark params)
    print("\nTraining RandomForest (n=200, depth=15)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    print(f"\n  F1-Score: {f1:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))

    # 7. Export
    print(f"\nExporting model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH, compress=3)
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  Model size: {model_size:.1f} MB")

    print(f"Exporting scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)

    print(f"Exporting feature columns to {FEATURES_PATH}...")
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\n" + "=" * 60)
    print("  EXPORT COMPLETE")
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Scaler:   {SCALER_PATH}")
    print(f"  Features: {FEATURES_PATH}")
    print("=" * 60)
    print("\nCopy the 'model/' directory to your Raspberry Pi to deploy.")


if __name__ == "__main__":
    main()
