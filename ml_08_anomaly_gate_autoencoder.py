#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__)))
TRAIN_PARQUET = os.path.join(BASE_DIR, "data", "train_data.parquet")
TEST_PARQUET = os.path.join(BASE_DIR, "data", "test_data.parquet")

JETSON_MODEL_DIR = os.path.join(BASE_DIR, "jetson", "model")
FEATURES_PATH = os.path.join(JETSON_MODEL_DIR, "feature_columns.json")

AE_MODEL_PATH = os.path.join(JETSON_MODEL_DIR, "anomaly_autoencoder.pkl")
AE_SCALER_PATH = os.path.join(JETSON_MODEL_DIR, "anomaly_scaler.pkl")
AE_THRESHOLD_PATH = os.path.join(JETSON_MODEL_DIR, "anomaly_threshold.json")


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
            "Run `python jetson/scripts/save_model.py` first to export `feature_columns.json`."
        )

    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)
    print(f"[OK] Loaded feature list: {len(feature_cols)} features")

    # Guard against train/serve skew: the exported feature set must already be
    # leakage-aware (no port columns). Fail loudly if a stale JSON sneaks them in.
    _LEAKY_PORTS = {"destination_port", "source_port", "src_port", "dst_port"}
    _leaky_in_json = [c for c in feature_cols if c.lower() in _LEAKY_PORTS]
    if _leaky_in_json:
        raise ValueError(
            f"feature_columns.json contains leaky port features {_leaky_in_json}. "
            "Re-export with the leakage-aware save_model.py before training the gate."
        )

    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"[OK] Loaded parquet: train={len(train_df):,}, test={len(test_df):,}")

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    train_benign = train_df[train_df["label_binary"] == 0]

    # ── Benign validation split for threshold selection ─────────────────────
    # Thresholds must NEVER be derived from the test distribution (that is an
    # oracle/peeking threshold). We hold out 10% of benign TRAINING flows as a
    # calibration set: the AE is fit on the remaining 90%, and every operating
    # threshold is a quantile of the VALIDATION benign MSE. The test set is
    # touched exactly once, to report metrics at those pre-committed thresholds.
    rng = np.random.RandomState(42)
    val_mask = rng.rand(len(train_benign)) < 0.10
    fit_benign = train_benign[~val_mask]
    val_benign = train_benign[val_mask]
    print(f"[OK] Benign split: fit={len(fit_benign):,}, "
          f"threshold-validation={len(val_benign):,} (10%, seed=42)")

    x_train = _safe_matrix(fit_benign, feature_cols)
    x_val = _safe_matrix(val_benign, feature_cols)
    x_test = _safe_matrix(test_df, feature_cols)
    y_test = test_df["label_binary"].values.astype(int)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
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

    # Thresholds from the held-out VALIDATION benign MSE (not fit-set MSE, which
    # is optimistically low; not test MSE, which would be an oracle threshold).
    val_hat = ae.predict(x_val_s)
    val_mse = _compute_mse(x_val_s, val_hat)

    default_quantile = 0.995
    threshold = float(np.quantile(val_mse, default_quantile))
    print(f"[OK] Threshold (q={default_quantile} on benign VALIDATION MSE): {threshold:.6f}")

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

    print("\n[Eval] Recall at target FPR (threshold from benign VALIDATION quantile,")
    print("       realised FPR measured on test — the deployable procedure):")
    for target_fpr in (0.01, 0.001):
        q = 1.0 - target_fpr
        thr = float(np.quantile(val_mse, q))
        m = _eval_at_threshold(y_test, test_mse, thr)
        print(f"  target FPR≈{target_fpr:.1%}: threshold={m['threshold']:.6f} | "
              f"realised FPR={m['fpr']:.4%} | Recall={m['recall']:.4f} | "
              f"Precision={m['precision']:.4f} | F1={m['f1']:.4f}")

    # Oracle reference ONLY (threshold from benign TEST quantile). This is NOT a
    # deployable procedure — it peeks at the test distribution — and exists only
    # to show how far validation-based calibration is from the oracle. Never
    # report these numbers as the gate's performance.
    print("\n[Eval] Oracle reference (test-quantile threshold — NOT deployable, "
          "for calibration-gap analysis only):")
    for target_fpr in (0.01, 0.001):
        q = 1.0 - target_fpr
        thr = float(np.quantile(benign_mse, q))
        m = _eval_at_threshold(y_test, test_mse, thr)
        print(f"  oracle @FPR≈{target_fpr:.1%}: Recall={m['recall']:.4f} "
              f"(vs. validation-calibrated above)")

    # ── Operating-point sweep ────────────────────────────────────────────────
    # The gate is a TUNABLE filter, not a single point. We sweep the threshold
    # over quantiles of the benign VALIDATION MSE (pre-committed before touching
    # test) and report, for each, the detection metrics AND the gate-skip ratio
    # (fraction of test flows the gate marks benign and thus removes from Spark
    # inference). This characterises the recall vs. offloading trade-off that a
    # single threshold hides. The operating point deployed on the edge must be
    # chosen from these validation quantiles, never re-tuned on test.
    results_dir = os.path.join(BASE_DIR, "results", "ml_08_anomaly_gate")
    os.makedirs(results_dir, exist_ok=True)
    sweep_q = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
    sweep_rows = []
    for q in sweep_q:
        thr = float(np.quantile(val_mse, q))
        m = _eval_at_threshold(y_test, test_mse, thr)
        gate_skip = float(np.mean(test_mse < thr))  # flows filtered as benign
        m.update({"val_quantile": q, "gate_skip_ratio": round(gate_skip, 4)})
        sweep_rows.append(m)
    sweep_df = pd.DataFrame(sweep_rows)[
        ["val_quantile", "threshold", "recall", "precision", "f1", "fpr", "gate_skip_ratio"]
    ]
    sweep_csv = os.path.join(results_dir, "gate_operating_points.csv")
    sweep_df.to_csv(sweep_csv, index=False)
    print("\n[Eval] Gate operating-point sweep (saved):")
    print(sweep_df.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(sweep_df["gate_skip_ratio"], sweep_df["recall"], "o-", color="#1f77b4", label="Recall")
        ax1.set_xlabel("Gate skip ratio (benign offloaded from Spark)")
        ax1.set_ylabel("Attack recall", color="#1f77b4")
        ax1.set_ylim(0, 1.12)
        ax2 = ax1.twinx()
        ax2.plot(sweep_df["gate_skip_ratio"], sweep_df["fpr"], "s--", color="#d62728", label="FPR")
        ax2.set_ylabel("False positive rate", color="#d62728")
        plt.title("Anomaly-gate operating points: recall vs. offload (and FPR)")
        fig.tight_layout()
        plot_path = os.path.join(results_dir, "gate_operating_points.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved: {plot_path}")
    except Exception as e:
        print(f"[WARN] Operating-point plot skipped: {e}")

    os.makedirs(JETSON_MODEL_DIR, exist_ok=True)
    joblib.dump(ae, AE_MODEL_PATH, compress=3)
    joblib.dump(scaler, AE_SCALER_PATH, compress=3)
    with open(AE_THRESHOLD_PATH, "w") as f:
        json.dump({
            "threshold": threshold,
            "quantile": default_quantile,
            "calibration": "benign_validation_split_10pct_seed42",
        }, f, indent=2)

    print("\n[OK] Exported anomaly artifacts:")
    print(f"  - {AE_MODEL_PATH}")
    print(f"  - {AE_SCALER_PATH}")
    print(f"  - {AE_THRESHOLD_PATH}")
    print("\nEdge enable:")
    print("  export ANOMALY_ENABLED=1")
    print("  python jetson/edge/kafka_consumer.py")


if __name__ == "__main__":
    main()

