#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Save Model — Export PySpark Pipeline for Raspberry Pi Deployment.

Runs on the training machine after experiments.  Trains a Decision Tree
pipeline on the SHAP Top-30 features and saves the full PySpark
PipelineModel (Assembler + Scaler + Classifier) to ``model/`` for
direct loading by the edge Prediction Engine.

Usage::

    python scripts/save_model.py

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import json
import shutil

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, THESIS_ROOT)

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    get_classifiers,
    Pipeline,
    VectorAssembler,
    StandardScaler,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "ids_pipeline_model")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# SHAP Top-30 features
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


def main():
    print("\n" + "=" * 60)
    print("  SAVE PYSPARK MODEL FOR RASPBERRY PI DEPLOYMENT")
    print("=" * 60 + "\n")

    # 1. Initialize Spark
    spark = create_spark_session("IDS_SaveModel")
    df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

    # 2. Select SHAP features
    selected_features = [f for f in SHAP_TOP_FEATURES if f in feature_cols]
    print(f"  Using {len(selected_features)} SHAP features")

    # 3. Build pipeline (Assembler + Scaler + Best Model)
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

    # Use Decision Tree - optimal for edge deployment:
    #   Model size: 0.026 MB (vs 2.848 MB for RF)
    #   F1-Score:   0.9962  (vs 0.9949 for RF)
    classifiers = get_classifiers(
        features_col="features_scaled",
        label_col="label_binary",
        num_features=len(selected_features),
    )
    best_model = classifiers["Decision Tree"]

    pipeline = Pipeline(stages=[assembler, scaler, best_model])

    print("\n  Training Decision Tree pipeline...")
    model = pipeline.fit(train_df)
    print("  [OK] Training complete")

    # 5. Quick evaluation
    from shared_utils import compute_metrics
    predictions = model.transform(test_df)
    metrics = compute_metrics(predictions)
    print(f"\n  Test F1-Score: {metrics['f1']:.6f}")
    print(f"  Test Accuracy: {metrics['accuracy']:.6f}")

    # 6. Save model
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

    model.save(MODEL_PATH)
    print(f"\n  [INFO] Model saved to: {MODEL_PATH}")

    # Calculate model size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(MODEL_PATH):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    print(f"  Model size: {total_size / (1024*1024):.1f} MB")

    # 7. Save feature columns
    with open(FEATURES_PATH, "w") as f:
        json.dump(selected_features, f, indent=2)
    print(f"  [INFO] Feature columns saved to: {FEATURES_PATH}")

    # 8. Done
    spark.stop()

    print("\n" + "=" * 60)
    print("  SAVE COMPLETE")
    print("=" * 60)
    print(f"\n  Copy to Raspberry Pi:")
    print(f"    scp -r {MODEL_DIR} pi@<rpi-ip>:~/raspberry/model/")
    print(f"\n  Model path on RPi: ./model/ids_pipeline_model")


if __name__ == "__main__":
    main()
