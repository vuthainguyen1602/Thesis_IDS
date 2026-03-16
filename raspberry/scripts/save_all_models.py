#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Save Multiple Models — Export DT, GBT, RF for RPi Benchmark Comparison.

Trains and saves three PySpark PipelineModels (Decision Tree, GBT,
Random Forest) with the SHAP Top-30 feature set.  Each model is saved
to a separate directory under ``model/`` for independent benchmarking.

Usage::

    python scripts/save_all_models.py

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

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

SHAP_TOP_FEATURES = [
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

MODELS_TO_SAVE = ["Decision Tree", "GBT", "Random Forest"]


def main():
    print("\n" + "=" * 60)
    print("  SAVE MULTIPLE MODELS FOR RPi BENCHMARK")
    print("=" * 60 + "\n")

    spark = create_spark_session("IDS_SaveAllModels")
    df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

    selected_features = [f for f in SHAP_TOP_FEATURES if f in feature_cols]
    print(f"  Using {len(selected_features)} SHAP features\n")

    # Save feature columns once
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

        # Evaluate
        predictions = model.transform(test_df)
        metrics = compute_metrics(predictions)

        # Save model
        safe_name = model_name.lower().replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"ids_pipeline_{safe_name}")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        model.save(model_path)

        # Model size
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

    # Summary
    print("\n" + "=" * 60)
    print("  ALL MODELS SAVED")
    print("=" * 60)
    print(f"\n  {'Model':<20} {'F1':>10} {'Size':>10} {'Train':>10}")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  {r['name']:<20} {r['f1']:>10.4f} {r['model_size_mb']:>8.3f}MB {r['train_time']:>8.1f}s")

    print(f"\n  Copy to RPi:")
    print(f"    scp -r {MODEL_DIR} pi@<rpi-ip>:~/raspberry/model/")

    # Save results JSON
    results_path = os.path.join(MODEL_DIR, "models_info.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    spark.stop()


if __name__ == "__main__":
    main()
