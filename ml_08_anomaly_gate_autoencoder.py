#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__)))
TRAIN_PARQUET = os.path.join(BASE_DIR, "data", "train_data.parquet")
TEST_PARQUET = os.path.join(BASE_DIR, "data", "test_data.parquet")

RPI_MODEL_DIR = os.path.join(BASE_DIR, "raspberry", "model")
FEATURES_PATH = os.path.join(RPI_MODEL_DIR, "feature_columns.json")

AE_MODEL_PATH = os.path.join(RPI_MODEL_DIR, "anomaly_autoencoder.pkl")
AE_SCALER_PATH = os.path.join(RPI_MODEL_DIR, "anomaly_scaler.pkl")
AE_THRESHOLD_PATH = os.path.join(RPI_MODEL_DIR, "anomaly_threshold.json")


def _safe_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].copy()
    x = x.fillna(0).replace([np.inf, -np.inf], 0)
    return x.values.astype(np.float32)


def _compute_mse(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    if x_hat.ndim == 1:
        x_hat = x_hat.reshape(-1, x.shape[1])
    return np.mean((x - x_hat) ** 2, axis=1)

def _eval_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
    }


def main():
    print("\n" + "=" * 70)
    print("  EXP 8: LIGHTWEIGHT AUTOENCODER ANOMALY DETECTION (EDGE)")
    print("=" * 70)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found.\n"
            "Run `python raspberry/scripts/save_model.py` first to export `feature_columns.json`."
        )

    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)
    print(f"[OK] Loaded feature list: {len(feature_cols)} features")

    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"[OK] Loaded parquet: train={len(train_df):,}, test={len(test_df):,}")

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    train_benign = train_df[train_df["label_binary"] == 0]
    print(f"[OK] Training AE on benign only: {len(train_benign):,} rows")

    x_train = _safe_matrix(train_benign, feature_cols)
    x_test = _safe_matrix(test_df, feature_cols)
    y_test = test_df["label_binary"].values.astype(int)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    ae = MLPRegressor(
        hidden_layer_sizes=(64, 16, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=4096,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=5,
        verbose=True,
    )
    ae.fit(x_train_s, x_train_s)

    train_hat = ae.predict(x_train_s)
    train_mse = _compute_mse(x_train_s, train_hat)

    default_quantile = 0.995
    threshold = float(np.quantile(train_mse, default_quantile))
    print(f"[OK] Threshold (q={default_quantile} on benign train MSE): {threshold:.6f}")

    test_hat = ae.predict(x_test_s)
    test_mse = _compute_mse(x_test_s, test_hat)

    test_benign = test_df[test_df["label_binary"] == 0]
    x_test_benign = _safe_matrix(test_benign, feature_cols)
    x_test_benign_s = scaler.transform(x_test_benign)
    benign_hat = ae.predict(x_test_benign_s)
    benign_mse = _compute_mse(x_test_benign_s, benign_hat)

    print("\n[Eval] Score-based metrics (higher = more anomalous):")
    try:
        roc = roc_auc_score(y_test, test_mse)
        print(f"  ROC-AUC: {roc:.6f}")
    except Exception as e:
        print(f"  ROC-AUC: [skip] {e}")
    try:
        ap = average_precision_score(y_test, test_mse)
        print(f"  PR-AUC:  {ap:.6f}")
    except Exception as e:
        print(f"  PR-AUC:  [skip] {e}")

    print("\n[Eval] Thresholded detection (Attack=1):")
    base = _eval_at_threshold(y_test, test_mse, threshold)
    print(f"  Default threshold: {base['threshold']:.6f}")
    print(f"    Precision: {base['precision']:.4f}")
    print(f"    Recall:    {base['recall']:.4f}")
    print(f"    F1:        {base['f1']:.4f}")
    print(f"    FPR:       {base['fpr']:.4%} (measured on full test)")

    print("\n[Eval] Recall at fixed FPR (threshold derived from BENIGN test distribution):")
    for target_fpr in (0.01, 0.001):
        q = 1.0 - target_fpr
        thr = float(np.quantile(benign_mse, q))
        m = _eval_at_threshold(y_test, test_mse, thr)
        print(f"  @FPR≈{target_fpr:.1%}: threshold={m['threshold']:.6f} | Recall={m['recall']:.4f} | Precision={m['precision']:.4f} | F1={m['f1']:.4f}")

    os.makedirs(RPI_MODEL_DIR, exist_ok=True)
    joblib.dump(ae, AE_MODEL_PATH, compress=3)
    joblib.dump(scaler, AE_SCALER_PATH, compress=3)
    with open(AE_THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold, "quantile": default_quantile}, f, indent=2)

    print("\n[OK] Exported anomaly artifacts:")
    print(f"  - {AE_MODEL_PATH}")
    print(f"  - {AE_SCALER_PATH}")
    print(f"  - {AE_THRESHOLD_PATH}")
    print("\nEdge enable:")
    print("  export ANOMALY_ENABLED=1")
    print("  python raspberry/edge/kafka_consumer.py")


if __name__ == "__main__":
    main()

