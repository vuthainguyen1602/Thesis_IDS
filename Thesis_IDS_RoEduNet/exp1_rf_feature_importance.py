#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
EXPERIMENT 1: RF FEATURE IMPORTANCE + FEATURE SELECTION (RoEduNet-SIMARGL2021)
================================================================================
Use Random Forest feature importance to select the most informative features,
then compare classification performance across different top-k values.
================================================================================
"""

import os
import pandas as pd
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
    export_multi_section_report,
    Pipeline,
    VectorAssembler,
    StandardScaler,
    RandomForestClassifier,
)

# ==============================================================================
# INITIALIZATION
# ==============================================================================
spark = create_spark_session("RoEduNet_Exp1_RF_Feature_Importance")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("█" * 70)
print("  EXPERIMENT 1: RF FEATURE IMPORTANCE (RoEduNet-SIMARGL2021)")
print("█" * 70)

# ==============================================================================
# STEP 1: COMPUTE RF FEATURE IMPORTANCE
# ==============================================================================
print("\n--- Step 1: Compute RF Feature Importance ---")

feature_importance_path = "/Users/thainguyenvu/Desktop/Thesis_IDS/Thesis_IDS_RoEduNet/feature_importance.csv"

try:
    feature_importance_df = pd.read_csv(feature_importance_path)
    print("✓ Read feature_importance.csv successfully")
except FileNotFoundError:
    print("Running RF to compute feature importance...")
    assembler_tmp = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
    scaler_tmp = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
    rf_tmp = RandomForestClassifier(
        featuresCol="features_scaled", labelCol="label_binary",
        numTrees=100, maxDepth=10, seed=42,
    )
    pipeline_tmp = Pipeline(stages=[assembler_tmp, scaler_tmp, rf_tmp])
    model_tmp = pipeline_tmp.fit(train_df)

    importances = model_tmp.stages[-1].featureImportances.toArray()
    feature_importance_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feature_importance_df = feature_importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print(f"✓ Saved feature importance to {feature_importance_path}")

print(f"\nTop-20 important features:")
print(feature_importance_df.head(20).to_string(index=True))

# ==============================================================================
# STEP 2: RUN WITH DIFFERENT TOP-K VALUES
# ==============================================================================
# RoEduNet has ~38 numeric features (vs 79 in CICIDS2017)
# So use smaller k values: [15, 20, 30]
top_k_values = [15, 20, 30]

all_section_results = []
base_output = "/Users/thainguyenvu/Desktop/Thesis_IDS/Thesis_IDS_RoEduNet/exp1_results"
os.makedirs(base_output, exist_ok=True)

report_sections = []

for k in top_k_values:
    print(f"\n\n{'█' * 70}")
    print(f"  RUNNING WITH TOP-{k} FEATURES")
    print(f"{'█' * 70}")

    selected_features = feature_importance_df.head(k)["feature"].tolist()
    print(f"  Selected: {selected_features}")

    assembler = VectorAssembler(inputCols=selected_features, outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)

    results, trained_models = run_all_classifiers(
        assembler=assembler,
        scaler=scaler,
        train_df=train_df,
        test_df=test_df,
        features_col="features_scaled",
        num_features=k,
    )

    # Ensemble Voting
    ens = ensemble_voting(trained_models, test_df, results=results)
    if ens:
        results["Ensemble Voting"] = ens

    print_summary_table(results, title=f"TOP-{k} FEATURES")

    plot_comparison(
        results,
        title=f"RoEduNet Exp 1: Top-{k} Features",
        save_path=os.path.join(base_output, f"exp1_top{k}_comparison.png"),
        show=False,
    )
    plot_training_time(
        results,
        title=f"RoEduNet Exp 1: Top-{k} Training Time",
        save_path=os.path.join(base_output, f"exp1_top{k}_train_time.png"),
        show=False,
    )
    plot_prediction_time(
        results,
        title=f"RoEduNet Exp 1: Top-{k} Prediction Time",
        save_path=os.path.join(base_output, f"exp1_top{k}_pred_time.png"),
        show=False,
    )
    plot_model_size(
        results,
        title=f"RoEduNet Exp 1: Top-{k} Model Size",
        save_path=os.path.join(base_output, f"exp1_top{k}_model_size.png"),
        show=False,
    )

    report_sections.append({
        "section_title": f"RF Feature Selection: Top-{k} Features",
        "results": results,
        "chart_paths": [
            os.path.join(base_output, f"exp1_top{k}_comparison.png"),
            os.path.join(base_output, f"exp1_top{k}_train_time.png"),
            os.path.join(base_output, f"exp1_top{k}_pred_time.png"),
            os.path.join(base_output, f"exp1_top{k}_model_size.png"),
        ],
    })

# ==============================================================================
# SUMMARY ACROSS ALL K VALUES
# ==============================================================================
print(f"\n\n{'█' * 70}")
print("  EXPERIMENT 1 SUMMARY: RF FEATURE IMPORTANCE (RoEduNet)")
print(f"{'█' * 70}")

for section in report_sections:
    k_name = section["section_title"]
    best_model = max(section["results"].items(), key=lambda x: x[1].get("f1", 0))
    print(f"  {k_name}: Best = {best_model[0]} (F1={best_model[1]['f1']:.6f})")

# Export HTML Report
export_multi_section_report(
    report_sections,
    title="RoEduNet - Experiment 1: RF Feature Importance",
    output_path=os.path.join(base_output, "exp1_report.html"),
)

print("\n✓ Experiment 1 (RoEduNet) completed!")
spark.stop()
print("✓ Spark Session closed.")
