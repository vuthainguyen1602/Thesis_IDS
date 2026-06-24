#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 11 — Cross-dataset generalization (distribution shift).

The robustness track in ml_07 is an *in-domain* proxy (a subsample of the same
test distribution). This script provides a true distribution-shift evaluation:
train on dataset A and test on dataset B (and the reverse), over the set of
leak-free features common to both. The gap between in-domain F1 (A->A) and
cross-domain F1 (A->B) quantifies how well the leakage-aware model transfers.

Both datasets must be prepared first with ml_00 (leakage-aware), e.g.:
    IDS_DATASET=cicids2017 IDS_DATA_DIR=$PWD/data_2017 IDS_RAW_DATA_DIR=... \\
        IDS_CSV_GLOB='*.csv' python ml_00_prepare_cicids2017.py
    IDS_DATASET=cicids2018 IDS_DATA_DIR=$PWD/data_2018 IDS_RAW_DATA_DIR=... \\
        IDS_CSV_GLOB='*.csv' python ml_00_prepare_cicids2017.py

Then point this script at the two prepared directories:
    IDS_XD_DIR_A=$PWD/data_2017 IDS_XD_NAME_A=CICIDS2017 \\
    IDS_XD_DIR_B=$PWD/data_2018 IDS_XD_NAME_B=CSE-CIC-IDS2018 \\
        python ml_11_cross_dataset_eval.py

Outputs (results/ml_11_cross_dataset/):
  - cross_dataset_results.csv   rows: train->test pairs; F1/precision/recall/AUC-PR
  - cross_dataset_f1.png        in-domain vs cross-domain F1, both directions
"""
import os

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    compute_metrics,
    ml_results_dir,
    Pipeline,
    VectorAssembler,
    StandardScaler,
    ML07_DIR,
)
from pyspark.ml.classification import RandomForestClassifier

OUT_DIR = os.path.join("results", "ml_11_cross_dataset")
os.makedirs(OUT_DIR, exist_ok=True)

DIR_A = os.environ.get("IDS_XD_DIR_A", os.path.join("data_2017"))
DIR_B = os.environ.get("IDS_XD_DIR_B", os.path.join("data_2018"))
NAME_A = os.environ.get("IDS_XD_NAME_A", "Dataset-A")
NAME_B = os.environ.get("IDS_XD_NAME_B", "Dataset-B")
RF_NUM_TREES = int(os.environ.get("IDS_XD_NUM_TREES", "200"))
RF_MAX_DEPTH = int(os.environ.get("IDS_XD_MAX_DEPTH", "15"))


def _fit(train_df, feature_cols):
    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                        handleInvalid="keep"),
        StandardScaler(inputCol="features_raw", outputCol="features",
                       withMean=True, withStd=True),
        RandomForestClassifier(featuresCol="features", labelCol="label_binary",
                               numTrees=RF_NUM_TREES, maxDepth=RF_MAX_DEPTH, seed=42),
    ])
    return pipeline.fit(train_df)


def _eval(model, test_df) -> dict:
    preds = model.transform(test_df).cache()
    preds.count()
    m = compute_metrics(preds)
    preds.unpersist()
    return m


def main():
    spark = create_spark_session("IDS_Exp11_CrossDataset")

    for d in (DIR_A, DIR_B):
        if not os.path.exists(os.path.join(d, "train_data.parquet")):
            raise FileNotFoundError(
                f"Prepared parquet not found in '{d}'. Run ml_00 for both datasets "
                "first (see this script's docstring)."
            )

    _, trainA, testA, featsA = load_and_prepare_data(spark, data_dir=DIR_A)
    _, trainB, testB, featsB = load_and_prepare_data(spark, data_dir=DIR_B)

    # Common leak-free feature set (both already exclude leaky port features),
    # preserving dataset-A column order for determinism.
    setB = set(featsB)
    common = [f for f in featsA if f in setB]
    print("\n" + "=" * 70)
    print("  EXPERIMENT 11: CROSS-DATASET GENERALIZATION")
    print("=" * 70)
    print(f"  A = {NAME_A} ({DIR_A}): {len(featsA)} features")
    print(f"  B = {NAME_B} ({DIR_B}): {len(featsB)} features")
    print(f"  Common leak-free features: {len(common)}")
    if not common:
        raise ValueError("No common features between the two datasets after "
                         "leakage-aware preparation.")

    keep = common + ["label_binary"]
    trainA, testA = trainA.select(keep), testA.select(keep)
    trainB, testB = trainB.select(keep), testB.select(keep)

    model_A = _fit(trainA, common)
    model_B = _fit(trainB, common)

    pairs = [
        (NAME_A, NAME_A, "in-domain", model_A, testA),
        (NAME_A, NAME_B, "cross",     model_A, testB),
        (NAME_B, NAME_B, "in-domain", model_B, testB),
        (NAME_B, NAME_A, "cross",     model_B, testA),
    ]
    rows = []
    for tr, te, kind, model, test_df in pairs:
        m = _eval(model, test_df)
        rows.append({
            "train": tr, "test": te, "kind": kind,
            "f1": m.get("f1"), "precision": m.get("precision"),
            "recall": m.get("recall"), "auc_pr": m.get("auc_pr"),
        })
        print(f"  {tr:>16} -> {te:<16} [{kind:9}] F1={m.get('f1')}")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "cross_dataset_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved: {csv_path}")
    print(df.to_string(index=False))

    # ── Figure: in-domain vs cross F1 for each train dataset ─────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        trains = [NAME_A, NAME_B]
        indom = [df[(df.train == t) & (df.kind == "in-domain")]["f1"].values[0] or 0
                 for t in trains]
        cross = [df[(df.train == t) & (df.kind == "cross")]["f1"].values[0] or 0
                 for t in trains]
        x = np.arange(len(trains)); w = 0.38
        fig, ax = plt.subplots(figsize=(7, 5))
        b1 = ax.bar(x - w / 2, indom, w, label="In-domain (A→A)", color="#2ca02c")
        b2 = ax.bar(x + w / 2, cross, w, label="Cross-dataset (A→B)", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels([f"Train: {t}" for t in trains])
        ax.set_ylim(0, 1.02); ax.set_ylabel("Binary F1")
        ax.set_title("Cross-dataset generalization (distribution shift)")
        ax.legend()
        ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
        fig.tight_layout()
        png_path = os.path.join(OUT_DIR, "cross_dataset_f1.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved: {png_path}")
    except Exception as e:
        print(f"[WARN] Cross-dataset plot skipped: {e}")

    spark.stop()


if __name__ == "__main__":
    main()
