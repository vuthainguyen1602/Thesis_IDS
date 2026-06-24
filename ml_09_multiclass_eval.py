#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 9 — Multiclass (per-attack) evaluation.

Binary IDS on CICIDS2017 is near-trivial: benign vs. attack is highly
separable, so every method scores F1 ≈ 1.0 and the four reduction strategies
look indistinguishable. The scientifically informative question is *per attack
class*: rare classes (Heartbleed, Infiltration, Web Attacks, Bot) are where
models actually fail and where feature-reduction strategies diverge.

This script trains the chosen reduction configuration in MULTICLASS mode and
reports per-class precision / recall / F1, macro- and weighted-F1, and the full
confusion matrix. It is additive — it does not modify the binary pipeline.

Run:
    python ml_09_multiclass_eval.py
Key env:
    IDS_MC_MODEL   = decision_tree | random_forest (default random_forest)
    IDS_MC_METHOD  = baseline | rf_top30 | shap_top30   (feature set, default baseline)
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    log_run_config,
    ml_results_dir,
    GLOBAL_SEED,
    ML02_DIR,
    ML06_DIR,
    ML09_DIR,
)

RF_IMPORTANCE_CSV = ml_results_dir(ML02_DIR, "feature_importance.csv", mkdir=False)
SHAP_IMPORTANCE_CSV = ml_results_dir(ML06_DIR, "shap_feature_importance.csv", mkdir=False)


def _select_features(method: str, feature_cols):
    """Resolve the feature set for the requested reduction method."""
    if method == "baseline":
        return feature_cols
    if method == "rf_top30":
        csv_path, col = RF_IMPORTANCE_CSV, "feature"
    elif method == "shap_top30":
        csv_path, col = SHAP_IMPORTANCE_CSV, "feature"
    else:
        raise ValueError(f"Unknown IDS_MC_METHOD: {method}")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found; falling back to baseline features.")
        return feature_cols
    ranked = pd.read_csv(csv_path).head(30)[col].tolist()
    # keep only columns that survive (e.g. after port exclusion)
    return [f for f in ranked if f in feature_cols]


def _per_class_metrics(pred_pdf: pd.DataFrame, labels):
    """Compute per-class precision/recall/F1/support from a label×prediction frame."""
    rows = []
    for idx, name in enumerate(labels):
        tp = int(((pred_pdf["label_idx"] == idx) & (pred_pdf["prediction"] == idx)).sum())
        fp = int(((pred_pdf["label_idx"] != idx) & (pred_pdf["prediction"] == idx)).sum())
        fn = int(((pred_pdf["label_idx"] == idx) & (pred_pdf["prediction"] != idx)).sum())
        support = int((pred_pdf["label_idx"] == idx).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append({
            "class": name, "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        })
    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)


def main():
    spark = create_spark_session("IDS_Exp9_Multiclass")
    method = os.environ.get("IDS_MC_METHOD", "baseline").strip().lower()
    model_name = os.environ.get("IDS_MC_MODEL", "random_forest").strip().lower()
    log_run_config("Exp9 Multiclass", extra={"IDS_MC_METHOD": method, "IDS_MC_MODEL": model_name})

    _, train_df, test_df, feature_cols = load_and_prepare_data(spark)
    out_dir = ml_results_dir(ML09_DIR)

    selected = _select_features(method, feature_cols)
    print(f"  Method={method} | features={len(selected)} | model={model_name}")

    # Multiclass label index from the original string label (preserved in parquet).
    indexer = StringIndexer(inputCol="label", outputCol="label_idx", handleInvalid="keep")
    assembler = VectorAssembler(inputCols=selected, outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled",
                            withStd=True, withMean=True)

    if model_name == "decision_tree":
        clf = DecisionTreeClassifier(featuresCol="features_scaled", labelCol="label_idx",
                                     maxDepth=15, impurity="entropy", seed=GLOBAL_SEED)
    else:
        clf = RandomForestClassifier(featuresCol="features_scaled", labelCol="label_idx",
                                     numTrees=200, maxDepth=15, featureSubsetStrategy="sqrt",
                                     seed=GLOBAL_SEED)

    pipeline = Pipeline(stages=[indexer, assembler, scaler, clf])
    print("  Training multiclass model ...")
    model = pipeline.fit(train_df)

    predictions = model.transform(test_df).select("label_idx", "prediction").cache()
    predictions.count()

    # Recover class names in index order from the fitted StringIndexer.
    labels = model.stages[0].labels
    print(f"  Classes ({len(labels)}): {labels}")

    ev = lambda metric: MulticlassClassificationEvaluator(
        labelCol="label_idx", predictionCol="prediction", metricName=metric
    ).evaluate(predictions)
    summary = {
        "method": method, "model": model_name,
        "n_classes": len(labels),
        "accuracy": round(ev("accuracy"), 6),
        "f1_weighted": round(ev("f1"), 6),
        "f1_macro": None,  # filled from per-class below
        "precision_weighted": round(ev("weightedPrecision"), 6),
        "recall_weighted": round(ev("weightedRecall"), 6),
    }

    pred_pdf = predictions.toPandas()
    predictions.unpersist()
    per_class = _per_class_metrics(pred_pdf, labels)
    # Macro-F1 over classes actually present in the test set: a train-only class
    # with zero test support would otherwise contribute an F1=0 and understate it.
    _present = per_class[per_class["support"] > 0]
    summary["f1_macro"] = round(float(_present["f1"].mean()), 6) if len(_present) else 0.0

    per_class_csv = os.path.join(out_dir, "per_class_metrics.csv")
    per_class.to_csv(per_class_csv, index=False)
    with open(os.path.join(out_dir, "multiclass_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Per-class metrics (sorted by support):")
    print(per_class.to_string(index=False))
    print("\n  Summary:", json.dumps(summary, indent=2))

    # Confusion matrix (row-normalised recall view).
    n = len(labels)
    cm = np.zeros((n, n), dtype=float)
    for _, r in pred_pdf.iterrows():
        cm[int(r["label_idx"]), int(r["prediction"])] += 1
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    plt.figure(figsize=(max(8, n), max(6, n)))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Row-normalised rate")
    plt.xticks(range(n), labels, rotation=90, fontsize=7)
    plt.yticks(range(n), labels, fontsize=7)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Multiclass Confusion ({method} / {model_name})")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[INFO] Saved: {per_class_csv}")
    print(f"[INFO] Saved: {cm_path}")
    print(f"[INFO] Results dir: {out_dir}")
    spark.stop()


if __name__ == "__main__":
    main()
