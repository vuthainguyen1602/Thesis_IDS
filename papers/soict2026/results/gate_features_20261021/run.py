#!/usr/bin/env python
"""Gate feature set: all 77 leak-free features vs. the classifier's SHAP Top-30.

Re-runs the gate's own training procedure (ml_08_anomaly_gate_autoencoder.py:
same benign 90/10 fit/validation split, seed 42, same autoencoder, threshold at
the 0.995 quantile of the validation benign MSE) once per feature set and scores
the CICIDS2017 test split. Nothing under jetson/model is written; the deployed
gate keeps the 77-feature model.

    python papers/soict2026/results/gate_features_20261021/run.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
import ml_08_anomaly_gate_autoencoder as g  # noqa: E402

QUANTILE = 0.995


def run(train_df, test_df, cols):
    benign = train_df[train_df["label_binary"] == 0]
    rng = np.random.RandomState(42)
    val_mask = rng.rand(len(benign)) < 0.10
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(g._safe_matrix(benign[~val_mask], cols))
    x_val = scaler.transform(g._safe_matrix(benign[val_mask], cols))
    x_test = scaler.transform(g._safe_matrix(test_df, cols))
    y_test = test_df["label_binary"].values.astype(int)
    ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", solver="adam",
                      alpha=1e-4, batch_size=4096, learning_rate_init=1e-3, max_iter=50,
                      random_state=42, early_stopping=True, validation_fraction=0.05,
                      n_iter_no_change=5)
    ae.fit(x_fit, x_fit)
    thr = float(np.quantile(g._compute_mse(x_val, ae.predict(x_val)), QUANTILE))
    mse = g._compute_mse(x_test, ae.predict(x_test))
    m = g._eval_at_threshold(y_test, mse, thr)
    m.update({"n_features": len(cols), "val_quantile": QUANTILE,
              "gate_skip_ratio": float(np.mean(mse < thr)), "ae_iterations": int(ae.n_iter_)})
    return m


def main():
    train_df = pd.read_parquet(g.TRAIN_PARQUET)
    test_df = pd.read_parquet(g.TEST_PARQUET)
    exclude = {"label", "label_binary", "source_ip", "destination_ip", "flow_id", "timestamp",
               "protocol", "destination_port", "source_port", "src_port", "dst_port"}
    all_cols = [c for c in train_df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
    shap = pd.read_csv(os.path.join(ROOT, "results", "ml_06_feature_selection_shap",
                                    "shap_feature_importance.csv"))
    shap30 = shap["feature"].head(30).tolist()
    deployed = json.load(open(os.path.join(ROOT, "jetson", "model", "anomaly_feature_columns.json")))
    assert sorted(deployed) == sorted(all_cols), "deployed gate list differs from the 77 leak-free features"

    rows = []
    for name, cols in (("all_leak_free_77", all_cols), ("shap_top30", shap30)):
        m = run(train_df, test_df, cols)
        m["feature_set"] = name
        rows.append(m)
        print(f"{name:18s} n={m['n_features']:2d} recall={m['recall']:.4f} "
              f"fpr={m['fpr']:.4%} skip={m['gate_skip_ratio']:.4f}", flush=True)
    out = pd.DataFrame(rows)[["feature_set", "n_features", "val_quantile", "threshold", "recall",
                              "fpr", "precision", "f1", "gate_skip_ratio", "tp", "fn", "fp", "tn",
                              "ae_iterations"]]
    out.to_csv(os.path.join(HERE, "gate_feature_sets.csv"), index=False)
    print(f"[OK] {os.path.join(HERE, 'gate_feature_sets.csv')}")


if __name__ == "__main__":
    main()
