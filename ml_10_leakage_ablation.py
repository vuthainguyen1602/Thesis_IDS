#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 10 — Feature-leakage ablation (destination_port).

Quantifies how much the label-leaking ``destination_port`` feature inflates
binary IDS performance. A single representative classifier (Random Forest) is
trained TWICE on the *same* train/test split:

  (A) WITHOUT port features  -> the leak-free, reported configuration
  (B) WITH    port features  -> the leaky configuration

Both feature sets come from the identical parquet split (the port columns are
retained in the parquet by ml_00; only feature selection differs), so the only
varying factor is the presence of the leaky feature.

Outputs (results/ml_10_leakage_ablation/):
  - leakage_ablation.csv   columns: config, port_included, f1, recall, precision, auc_pr
  - leakage_ablation.png   grouped bar chart (with vs without port)

These feed FAIR'2026 Table tab:port-ablation and the leakage-ablation figure.
This script is additive: it does not modify any other experiment.
"""
import os
import pandas as pd

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    Pipeline,
    VectorAssembler,
    StandardScaler,
)
from idslib.data import _leaky_port_cols
from idslib.modeling import train_and_evaluate
from pyspark.ml.classification import RandomForestClassifier

OUT_DIR = os.path.join("results", "ml_10_leakage_ablation")
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed, a-priori model so the comparison is purely about the feature set.
RF_NUM_TREES = int(os.environ.get("IDS_ABLATION_NUM_TREES", "100"))
RF_MAX_DEPTH = int(os.environ.get("IDS_ABLATION_MAX_DEPTH", "10"))


def _build_pipeline(feature_cols):
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True,
    )
    clf = RandomForestClassifier(
        featuresCol="features", labelCol="label_binary",
        numTrees=RF_NUM_TREES, maxDepth=RF_MAX_DEPTH, seed=42,
    )
    return Pipeline(stages=[assembler, scaler, clf])


def _run(train_df, test_df, feature_cols, title):
    pipeline = _build_pipeline(feature_cols)
    _model, _preds, metrics = train_and_evaluate(pipeline, train_df, test_df, title)
    return metrics


def main():
    spark = create_spark_session("IDS_Exp10_Leakage_Ablation")
    # Loaded with port EXCLUDED (default). We re-add the port columns manually
    # for the leaky run so both runs share the exact same split.
    _df, train_df, test_df, feat_noport = load_and_prepare_data(spark)

    present_ports = [c for c in _leaky_port_cols() if c in train_df.columns]
    feat_withport = feat_noport + present_ports

    print("\n" + "=" * 70)
    print("  EXPERIMENT 10: FEATURE-LEAKAGE ABLATION (destination_port)")
    print("=" * 70)
    print(f"  Features without port: {len(feat_noport)}")
    print(f"  Port columns re-added : {present_ports if present_ports else 'NONE FOUND'}")
    print(f"  Features with port    : {len(feat_withport)}")
    print(f"  Model: RandomForest(numTrees={RF_NUM_TREES}, maxDepth={RF_MAX_DEPTH})")

    if not present_ports:
        print("  [WARN] No port columns present in the parquet — cannot run the "
              "leaky arm. Re-run ml_00 without dropping port columns.")

    rows = []
    m_no = _run(train_df, test_df, feat_noport,
                "Leak-free (no destination_port)")
    rows.append({"config": "Without destination_port (proposed)",
                 "port_included": False, **{k: m_no.get(k) for k in
                 ("f1", "recall", "precision", "auc_pr")}})

    if present_ports:
        m_with = _run(train_df, test_df, feat_withport,
                      "Leaky (with destination_port)")
        rows.append({"config": "With destination_port (leaky)",
                     "port_included": True, **{k: m_with.get(k) for k in
                     ("f1", "recall", "precision", "auc_pr")}})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "leakage_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved: {csv_path}")
    print(df.to_string(index=False))

    # ── Bar chart (with vs without port) ─────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metrics_keys = ["f1", "recall", "precision", "auc_pr"]
        labels = ["F1", "Recall", "Precision", "AUC-PR"]
        x = np.arange(len(metrics_keys))
        width = 0.38

        fig, ax = plt.subplots(figsize=(8, 5))
        no_vals = [df.loc[~df.port_included, k].values[0] if (~df.port_included).any()
                   else 0 for k in metrics_keys]
        ax.bar(x - width / 2, [v if v is not None else 0 for v in no_vals], width,
               label="Without port (proposed)", color="#2ca02c")
        if df.port_included.any():
            with_vals = [df.loc[df.port_included, k].values[0] for k in metrics_keys]
            ax.bar(x + width / 2, [v if v is not None else 0 for v in with_vals], width,
                   label="With port (leaky)", color="#d62728")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Score (binary)")
        ax.set_title("Feature-leakage ablation: effect of destination_port")
        ax.legend()
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
        fig.tight_layout()
        png_path = os.path.join(OUT_DIR, "leakage_ablation.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved: {png_path}")
    except Exception as e:
        print(f"[WARN] Ablation plot skipped: {e}")

    spark.stop()


if __name__ == "__main__":
    main()
