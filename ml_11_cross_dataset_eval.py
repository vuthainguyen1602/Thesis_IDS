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


def _with_class_weights(train_df):
    """Add inverse-frequency class weights, matching the uniform imbalance
    handling (weightCol) declared for all weight-aware models in the paper —
    the cross-dataset RF must not silently train unweighted."""
    from pyspark.sql import functions as F
    counts = {int(r["label_binary"]): r["count"]
              for r in train_df.groupBy("label_binary").count().collect()}
    total = sum(counts.values())
    n_classes = max(len(counts), 1)
    weights = {lbl: total / (n_classes * c) for lbl, c in counts.items() if c > 0}
    expr = F.when(F.col("label_binary") == 1, float(weights.get(1, 1.0))) \
            .otherwise(float(weights.get(0, 1.0)))
    print(f"  Class weights (inverse frequency): {weights}")
    return train_df.withColumn("class_weight", expr)


def _fit(train_df, feature_cols):
    train_df = _with_class_weights(train_df)
    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                        handleInvalid="keep"),
        StandardScaler(inputCol="features_raw", outputCol="features",
                       withMean=True, withStd=True),
        RandomForestClassifier(featuresCol="features", labelCol="label_binary",
                               weightCol="class_weight",
                               numTrees=RF_NUM_TREES, maxDepth=RF_MAX_DEPTH, seed=42,
                               # Cap the per-iteration node-stats aggregation buffer:
                               # the default 256MB OOMs 4GB Jetson executors in
                               # findBestSplits on the ~1.9M-row IDS2018 fit.
                               maxMemoryInMB=int(os.environ.get("IDS_XD_MAX_MEMORY_MB", "128"))),
    ])
    return pipeline.fit(train_df)


def _eval(model, test_df) -> dict:
    preds = model.transform(test_df).cache()
    preds.count()
    m = compute_metrics(preds)
    preds.unpersist()
    return m


# ── Unsupervised domain adaptation baselines ─────────────────────────────────
# The cross-dataset rows above measure the gap. These two measure whether the
# cheapest treatments close any of it, WITHOUT touching the trained classifier
# and WITHOUT any label from the target domain — the only setting that is
# honest for a detector already deployed on a new network.
#
#   scaler-refit : the source pipeline standardises the target with the SOURCE
#                  mean/std. Since the two testbeds ran different CICFlowMeter
#                  versions, part of the collapse may be scale mismatch rather
#                  than a genuine shift. Refit only the StandardScaler on the
#                  target's (unlabelled) training features.
#   coral        : also align second-order statistics — whiten the target with
#                  its own covariance, then recolour it with the source's, so
#                  the classifier sees inputs shaped like what it was fitted on
#                  (Sun et al., "Return of Frustratingly Easy Domain
#                  Adaptation", AAAI 2016).
ADAPT_ENABLED = os.environ.get("IDS_XD_ADAPT", "1") == "1"


def _assembled(df, feature_cols):
    return VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                           handleInvalid="keep").transform(df)


def _mean_cov(df_assembled):
    """Mean vector and covariance matrix of the assembled feature column."""
    import numpy as np
    from pyspark.mllib.linalg import Vectors as MLlibVectors
    from pyspark.mllib.linalg.distributed import RowMatrix

    rdd = df_assembled.select("features_raw").rdd.map(
        lambda r: MLlibVectors.dense(r[0].toArray()))
    rm = RowMatrix(rdd)
    mean = np.array(rm.computeColumnSummaryStatistics().mean())
    cov = np.array(rm.computeCovariance().toArray())
    return mean, cov


def _sqrt_psd(mat, inverse=False, rel_eps=1e-8):
    """Symmetric PSD (inverse) square root, with a RELATIVE eigenvalue floor.

    An absolute floor is useless here: these features span fourteen orders of
    magnitude (byte counts against ratios), so a 1e-6 cut leaves near-null
    directions in place and the inverse root amplifies them by 1e8. Flooring at
    a fraction of the largest eigenvalue keeps the map conditioned.
    """
    import numpy as np
    vals, vecs = np.linalg.eigh(mat)
    floor = max(float(vals.max()), 0.0) * rel_eps
    vals = np.clip(vals, floor if floor > 0 else 1e-12, None)
    vals = 1.0 / np.sqrt(vals) if inverse else np.sqrt(vals)
    return (vecs * vals) @ vecs.T


def _coral_map(src_stats, tgt_stats):
    """Linear map sending TARGET features into the SOURCE feature space.

    The alignment runs on CORRELATION matrices rather than raw covariances:
    whitening a covariance whose diagonal spans 1e14 is numerically hopeless,
    while correlations have a unit diagonal and are well conditioned. Scale is
    restored afterwards with each domain's own standard deviations, so the
    result is still the CORAL map — second-order alignment plus mean shift.
    """
    import numpy as np
    mu_s, cov_s = src_stats
    mu_t, cov_t = tgt_stats
    sd_s = np.sqrt(np.clip(np.diag(cov_s), 0, None))
    sd_t = np.sqrt(np.clip(np.diag(cov_t), 0, None))
    # Constant features carry no information to align; map them through as-is.
    sd_s_safe = np.where(sd_s > 0, sd_s, 1.0)
    sd_t_safe = np.where(sd_t > 0, sd_t, 1.0)
    corr_s = cov_s / np.outer(sd_s_safe, sd_s_safe)
    corr_t = cov_t / np.outer(sd_t_safe, sd_t_safe)
    align = _sqrt_psd(corr_t, inverse=True) @ _sqrt_psd(corr_s)
    W = (align / sd_t_safe[:, None]) * sd_s_safe[None, :]
    return mu_s, mu_t, W


def _apply_linear_map(df_assembled, mu_s, mu_t, W):
    import numpy as np
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import Vectors, VectorUDT

    # Bind the arrays once: rebuilding them per row would dominate the cost of
    # what is otherwise a 60x60 mat-vec.
    a_mu_s = np.asarray(mu_s, dtype=float)
    a_mu_t = np.asarray(mu_t, dtype=float)
    a_W = np.asarray(W, dtype=float)

    def _map(v):
        return Vectors.dense((np.asarray(v.toArray(), dtype=float) - a_mu_t) @ a_W + a_mu_s)

    return df_assembled.withColumn("features_raw", udf(_map, VectorUDT())("features_raw"))


def _eval_adapted(model, test_assembled):
    """Run the fitted scaler + classifier over an already-assembled frame."""
    from pyspark.ml import PipelineModel

    tail = PipelineModel(stages=[st for st in model.stages
                                 if "VectorAssembler" not in type(st).__name__])
    preds = tail.transform(test_assembled).cache()
    preds.count()
    m = compute_metrics(preds)
    preds.unpersist()
    return m


def _eval_scaler_refit(model, target_train, target_test, feature_cols):
    """Keep the classifier, restandardise with the target's own statistics."""
    from pyspark.ml import PipelineModel

    scaler_fit = StandardScaler(inputCol="features_raw", outputCol="features",
                                withMean=True, withStd=True).fit(
        _assembled(target_train, feature_cols))
    clf = [st for st in model.stages if "Classification" in type(st).__name__][0]
    preds = PipelineModel(stages=[scaler_fit, clf]).transform(
        _assembled(target_test, feature_cols)).cache()
    preds.count()
    m = compute_metrics(preds)
    preds.unpersist()
    return m


def _run_adaptation(model, source_train, target_train, target_test, feature_cols, tag):
    """Both baselines for one direction; returns rows ready for the CSV."""
    rows = []
    m = _eval_scaler_refit(model, target_train, target_test, feature_cols)
    rows.append({"adaptation": "scaler-refit (target statistics)", **m})
    print(f"  {tag} | scaler-refit : F1={m.get('f1')}")

    src_stats = _mean_cov(_assembled(source_train, feature_cols))
    tgt_stats = _mean_cov(_assembled(target_train, feature_cols))
    mu_s, mu_t, W = _coral_map(src_stats, tgt_stats)
    mapped = _apply_linear_map(_assembled(target_test, feature_cols), mu_s, mu_t, W)
    m = _eval_adapted(model, mapped)
    rows.append({"adaptation": "CORAL (target -> source)", **m})
    print(f"  {tag} | CORAL        : F1={m.get('f1')}")
    return rows


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
    # preserving dataset-A column order for determinism. Column names of both
    # datasets are harmonised to the CICIDS2017 canonical vocabulary at
    # preparation time (idslib.data.CICIDS2018_TO_2017_ALIASES), so this
    # intersection compares semantics, not spelling variants.
    setB = set(featsB)
    common = [f for f in featsA if f in setB]
    only_a = sorted(set(featsA) - setB)
    only_b = sorted(setB - set(featsA))
    print("\n" + "=" * 70)
    print("  EXPERIMENT 11: CROSS-DATASET GENERALIZATION")
    print("=" * 70)
    print(f"  A = {NAME_A} ({DIR_A}): {len(featsA)} features")
    print(f"  B = {NAME_B} ({DIR_B}): {len(featsB)} features")
    print(f"  Common leak-free features: {len(common)}")
    if only_a:
        print(f"  Only in A ({len(only_a)}): {', '.join(only_a[:10])}"
              f"{'...' if len(only_a) > 10 else ''}")
    if only_b:
        print(f"  Only in B ({len(only_b)}): {', '.join(only_b[:10])}"
              f"{'...' if len(only_b) > 10 else ''}")
    if not common:
        raise ValueError("No common features between the two datasets after "
                         "leakage-aware preparation.")
    if len(common) < 0.5 * min(len(featsA), len(featsB)):
        print(f"  [WARN] Common feature set ({len(common)}) is under half of the "
              "smaller dataset's features — check that ml_00 was re-run for BOTH "
              "datasets AFTER the 2018->2017 column-alias harmonisation, "
              "otherwise the intersection is on stale un-harmonised parquet.")

    # Persist the mapping/intersection for the paper (reviewers must be able to
    # verify which features the cross-dataset comparison actually used).
    with open(os.path.join(OUT_DIR, "common_features.txt"), "w") as f:
        f.write("\n".join(common) + "\n")
    print(f"  [INFO] Common feature list saved -> {os.path.join(OUT_DIR, 'common_features.txt')}")

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

    # ── Domain adaptation (separate artefact; the rows above stay as published)
    if ADAPT_ENABLED:
        print("\n  Unsupervised domain adaptation (classifier untouched, no target labels):")
        adapt_rows = []
        for src, tgt, model, src_train, tgt_train, tgt_test in (
            (NAME_A, NAME_B, model_A, trainA, trainB, testB),
            (NAME_B, NAME_A, model_B, trainB, trainA, testA),
        ):
            baseline = next((r for r in rows if r["train"] == src and r["test"] == tgt), {})
            for r in _run_adaptation(model, src_train, tgt_train, tgt_test, common,
                                     f"{src} -> {tgt}"):
                adapt_rows.append({
                    "train": src, "test": tgt,
                    "adaptation": r["adaptation"],
                    "f1": r.get("f1"), "precision": r.get("precision"),
                    "recall": r.get("recall"), "auc_pr": r.get("auc_pr"),
                    "f1_no_adaptation": baseline.get("f1"),
                    "delta_f1": (r.get("f1") or 0.0) - (baseline.get("f1") or 0.0),
                })
        if adapt_rows:
            import pandas as _pd
            adapt_df = _pd.DataFrame(adapt_rows)
            adapt_path = os.path.join(OUT_DIR, "cross_dataset_adaptation.csv")
            adapt_df.to_csv(adapt_path, index=False)
            print(f"\n[INFO] Saved: {adapt_path}")
            print(adapt_df.to_string(index=False))

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
        ax.set_ylim(0, 1.12); ax.set_ylabel("Binary F1")
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
