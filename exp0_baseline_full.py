#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 0 — Baseline Evaluation Using All Features.

Trains every classifier on the full CICIDS2017 feature set (no feature
selection or dimensionality reduction) to establish baseline performance.
Results serve as the reference point for subsequent experiments.

Algorithms evaluated: DT, LR, SVM, NB, RF, GBT, XGBoost, LightGBM, MLP,
Bagging Ensemble, Ensemble Voting.

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import time
from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    run_all_classifiers,
    ensemble_voting,
    plot_comparison,
    plot_training_time,
    plot_prediction_time,
    plot_model_size,
    plot_confusion_matrices,
    plot_roc_curves,
    print_summary_table,
    export_multi_section_report,
    Pipeline,
    VectorAssembler,
    StandardScaler,
)

# ==============================================================================
# INITIALIZATION
# ==============================================================================
spark = create_spark_session("IDS_Exp0_Baseline_Full")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("=" * 70)
print("  EXPERIMENT 0: BASELINE - FULL FEATURE SET")
print("=" * 70)
print(f"  Total Features: {len(feature_cols)}")

# Create output directory
base_output = "/Users/thainguyenvu/Desktop/Thesis_IDS/exp0_results"
os.makedirs(base_output, exist_ok=True)

# ==============================================================================
# STEP 1: RUN ALL CLASSIFIERS
# ==============================================================================
# Configure Assembler and Scaler for ALL features
assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep",
)
scaler = StandardScaler(
    inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
)

# run_all_classifiers includes:
# - Standalone models (DT, LR, SVM, NB, RF, GBT, XGB, LGBM, MLP)
# - The new Bagging Ensemble (10x Best Model)
results, trained_models = run_all_classifiers(
    assembler=assembler,
    scaler=scaler,
    train_df=train_df,
    test_df=test_df,
    features_col="features_scaled",
    num_features=len(feature_cols),
)

# Step 2: Majority Voting Ensemble
print("\n--- Running Ensemble Voting (Majority Voting) ---")
ens_metrics = ensemble_voting(trained_models, test_df, results=results)
if ens_metrics:
    results["Ensemble Voting"] = ens_metrics

# ==============================================================================
# VISUALIZATION & REPORTING
# ==============================================================================
# Results table
print_summary_table(results, title="RESULTS: BASELINE (ALL FEATURES)")

# Plotting
plot_comparison(
    results,
    title="Experiment 0: Baseline All-Features Comparison",
    save_path=os.path.join(base_output, "comparison.png"),
    show=False,
)
plot_training_time(
    results,
    title="Experiment 0: Baseline Training Time",
    save_path=os.path.join(base_output, "train_time.png"),
    show=False,
)
plot_prediction_time(
    results,
    title="Experiment 0: Baseline Prediction Time",
    save_path=os.path.join(base_output, "pred_time.png"),
    show=False,
)
plot_model_size(
    results,
    title="Experiment 0: Baseline Model Size",
    save_path=os.path.join(base_output, "model_size.png"),
    show=False,
)
plot_confusion_matrices(
    results,
    title="Experiment 0: Confusion Matrices",
    save_path=os.path.join(base_output, "confusion_matrices.png"),
    show=False,
)
plot_roc_curves(
    trained_models, test_df,
    title="Experiment 0: ROC Curves",
    save_path=os.path.join(base_output, "roc_curves.png"),
    show=False,
)

# Export Comprehensive HTML Report
report_sections = [{
    "section_title": "Baseline Performance Evaluation (Full Features)",
    "results": results,
    "chart_paths": [
        os.path.join(base_output, "comparison.png"),
        os.path.join(base_output, "train_time.png"),
        os.path.join(base_output, "pred_time.png"),
        os.path.join(base_output, "model_size.png"),
        os.path.join(base_output, "confusion_matrices.png"),
        os.path.join(base_output, "roc_curves.png"),
    ]
}]

export_multi_section_report(
    report_sections, 
    title="IDS Thesis - Experiment 0: Baseline All-Features Evaluation",
    output_path=os.path.join(base_output, "exp0_report.html")
)

print("\n[INFO] Experiment 0 completed!")
print(f"[INFO] Results exported to: {base_output}")
spark.stop()
print("[INFO] Spark Session closed.")
