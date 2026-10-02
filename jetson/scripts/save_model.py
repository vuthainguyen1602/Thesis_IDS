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
    Pipeline,
    VectorAssembler,
    StandardScaler,
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "ids_pipeline_model")
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


def main():
    os.environ.setdefault("IDS_ALLOW_LOCAL_SPARK", "1")
    print("\n" + "=" * 60)
    print("  SAVE PYSPARK MODEL FOR JETSON DEPLOYMENT")
    print("=" * 60 + "\n")

    spark = create_spark_session("IDS_SaveModel")
    _df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

    selected_features = [f for f in SHAP_TOP_FEATURES if f in feature_cols]
    print(f"  Using {len(selected_features)} SHAP features")

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
    best_model = classifiers["Random Forest"]

    pipeline = Pipeline(stages=[assembler, scaler, best_model])

    print("\n  Training Random Forest pipeline...")
    from shared_utils import add_class_weights
    train_df = add_class_weights(train_df)  # weightCol-aware model needs this
    model = pipeline.fit(train_df)
    print("  [OK] Training complete")

    from shared_utils import compute_metrics
    predictions = model.transform(test_df)
    metrics = compute_metrics(predictions)
    print(f"\n  Test F1-Score: {metrics['f1']:.6f}")
    print(f"  Test Accuracy: {metrics['accuracy']:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

    model.save(MODEL_PATH)
    print(f"\n  [INFO] Model saved to: {MODEL_PATH}")

    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(MODEL_PATH):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    print(f"  Model size: {total_size / (1024*1024):.1f} MB")

    with open(FEATURES_PATH, "w") as f:
        json.dump(selected_features, f, indent=2)
    print(f"  [INFO] Feature columns saved to: {FEATURES_PATH}")

    spark.stop()

    print("\n" + "=" * 60)
    print("  SAVE COMPLETE")
    print("=" * 60)
    print(f"\n  Copy to Jetson:")
    print(f"    scp -r {MODEL_DIR}/* <user>@<jetson-ip>:~/Thesis_IDS/jetson/model/")
    print(f"\n  Model path on the Jetson: ~/Thesis_IDS/jetson/model/ids_pipeline_model")


if __name__ == "__main__":
    main()
