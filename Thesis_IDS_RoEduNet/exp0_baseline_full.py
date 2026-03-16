#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
EXPERIMENT 0: BASELINE - ALL FEATURES (RoEduNet-SIMARGL2021)
================================================================================
Train all classifiers on the FULL feature set (no feature selection).
This serves as the baseline for comparison with other experiments.
================================================================================
"""

import os
from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    run_all_classifiers,
    ensemble_voting,
    plot_comparison,
    plot_training_time,
    plot_prediction_time,
    plot_model_size,
    print_summary_table,
    export_results_to_html,
    VectorAssembler,
    StandardScaler,
)

# ==============================================================================
# INITIALIZATION
# ==============================================================================
spark = create_spark_session("RoEduNet_Exp0_Baseline")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("█" * 70)
print("  EXPERIMENT 0: BASELINE - ALL FEATURES (RoEduNet-SIMARGL2021)")
print("█" * 70)
print(f"  Dataset: RoEduNet-SIMARGL2021")
print(f"  Number of features: {len(feature_cols)}")

# ==============================================================================
# STEP 1: PREPARE PIPELINE
# ==============================================================================
assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep"
)
scaler = StandardScaler(
    inputCol="features_raw", outputCol="features_scaled",
    withStd=True, withMean=True,
)

# ==============================================================================
# STEP 2: RUN ALL CLASSIFIERS
# ==============================================================================
results, trained_models = run_all_classifiers(
    assembler=assembler,
    scaler=scaler,
    train_df=train_df,
    test_df=test_df,
    features_col="features_scaled",
    num_features=len(feature_cols),
)

# Ensemble Voting
ens_metrics = ensemble_voting(trained_models, test_df, results=results)
if ens_metrics:
    results["Ensemble Voting"] = ens_metrics

# ==============================================================================
# RESULTS
# ==============================================================================
print_summary_table(results, title="EXPERIMENT 0 SUMMARY: BASELINE (RoEduNet)")

base_output = "/Users/thainguyenvu/Desktop/Thesis_IDS/Thesis_IDS_RoEduNet/exp0_results"
os.makedirs(base_output, exist_ok=True)

plot_comparison(
    results,
    title="RoEduNet Exp 0: Baseline - All Features",
    save_path=os.path.join(base_output, "exp0_comparison.png"),
    show=False,
)
plot_training_time(
    results,
    title="RoEduNet Exp 0: Training Time",
    save_path=os.path.join(base_output, "exp0_train_time.png"),
    show=False,
)
plot_prediction_time(
    results,
    title="RoEduNet Exp 0: Prediction Time",
    save_path=os.path.join(base_output, "exp0_pred_time.png"),
    show=False,
)
plot_model_size(
    results,
    title="RoEduNet Exp 0: Model Size",
    save_path=os.path.join(base_output, "exp0_model_size.png"),
    show=False,
)

export_results_to_html(
    results,
    title="RoEduNet - Experiment 0: Baseline",
    output_path=os.path.join(base_output, "exp0_report.html"),
    chart_paths=[
        os.path.join(base_output, "exp0_comparison.png"),
        os.path.join(base_output, "exp0_train_time.png"),
        os.path.join(base_output, "exp0_pred_time.png"),
        os.path.join(base_output, "exp0_model_size.png"),
    ],
)

print("\n✓ Experiment 0 (RoEduNet Baseline) completed!")
spark.stop()
print("✓ Spark Session closed.")
