#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, THESIS_ROOT)

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    get_classifiers,
    compute_metrics,
    Pipeline,
    VectorAssembler,
    StandardScaler,
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

def _shap_top_features(k: int = 30) -> list:
    """Leakage-free SHAP Top-k, read from the ml_06 ranking so the exported model
    cannot drift from the feature set the offline study selected."""
    import csv
    path = os.path.join(THESIS_ROOT, "results", "ml_06_feature_selection_shap",
                        "shap_feature_importance.csv")
    leaky = {"destination_port", "source_port", "src_port", "dst_port"}
    with open(path) as fh:
        ranked = [r["feature"] for r in csv.DictReader(fh)]
    return [f for f in ranked if f not in leaky][:k]


SHAP_TOP_FEATURES = _shap_top_features()

MODELS_TO_SAVE = ["Decision Tree", "GBT", "Random Forest"]


def main():
    print("\n" + "=" * 60)
    print("  SAVE MULTIPLE MODELS FOR JETSON BENCHMARK")
    print("=" * 60 + "\n")

    spark = create_spark_session("IDS_SaveAllModels")
    _df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

    # These models are exported ONLY to benchmark edge inference (latency /
    # throughput / size). Model structure (e.g. RF numTrees/maxDepth) — and thus
    # inference cost — is unchanged by the training-set size, so we may train on a
    # stratified subsample to keep a single-JVM save within memory. Accuracy for
    # the thesis comes from the distributed ml_07 run, not from these artifacts.
    _frac = float(os.environ.get("IDS_EXPORT_SAMPLE_FRAC", "0.2"))
    if 0.0 < _frac < 1.0:
        label_col = "label_binary"
        fractions = {r[label_col]: _frac for r in train_df.select(label_col).distinct().collect()}
        train_df = train_df.sampleBy(label_col, fractions=fractions, seed=42)
        print(f"  [export] Stratified subsample for save: frac={_frac} "
              f"-> {train_df.count():,} train rows (structure/inference cost unchanged)")

    selected_features = [f for f in SHAP_TOP_FEATURES if f in feature_cols]
    print(f"  Using {len(selected_features)} SHAP features\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(FEATURES_PATH, "w") as f:
        json.dump(selected_features, f, indent=2)

    assembler = VectorAssembler(
        inputCols=selected_features,
        outputCol="features_raw",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withStd=True,
        withMean=True,
    )

    classifiers = get_classifiers(
        features_col="features_scaled",
        label_col="label_binary",
        num_features=len(selected_features),
    )

    from shared_utils import add_class_weights
    train_df = add_class_weights(train_df)  # weightCol-aware models need this

    results = []

    for model_name in MODELS_TO_SAVE:
        print(f"\n{'─' * 50}")
        print(f"  Training: {model_name}")
        print(f"{'─' * 50}")

        classifier = classifiers[model_name]
        pipeline = Pipeline(stages=[assembler, scaler, classifier])

        import time
        start = time.time()
        model = pipeline.fit(train_df)
        train_time = time.time() - start

        predictions = model.transform(test_df)
        metrics = compute_metrics(predictions)

        safe_name = model_name.lower().replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"ids_pipeline_{safe_name}")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        model.save(model_path)

        total_size = 0
        for dirpath, _, filenames in os.walk(model_path):
            for fn in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, fn))

        info = {
            "name": model_name,
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "train_time": train_time,
            "model_size_mb": total_size / (1024 * 1024),
            "path": model_path,
        }
        results.append(info)

        print(f"  F1: {metrics['f1']:.6f} | Acc: {metrics['accuracy']:.6f}")
        print(f"  Train: {train_time:.1f}s | Size: {info['model_size_mb']:.3f} MB")
        print(f"  Saved: {model_path}")

    print("\n" + "=" * 60)
    print("  ALL MODELS SAVED")
    print("=" * 60)
    print(f"\n  {'Model':<20} {'F1':>10} {'Size':>10} {'Train':>10}")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  {r['name']:<20} {r['f1']:>10.4f} {r['model_size_mb']:>8.3f}MB {r['train_time']:>8.1f}s")

    print(f"\n  Copy to Jetson:")
    print(f"    scp -r {MODEL_DIR}/* <user>@<jetson-ip>:~/Thesis_IDS/jetson/model/")

    results_path = os.path.join(MODEL_DIR, "models_info.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    spark.stop()


if __name__ == "__main__":
    main()
