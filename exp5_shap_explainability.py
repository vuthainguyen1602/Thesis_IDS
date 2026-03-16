#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 5 — SHAP Explainability (XAI for IDS).

Uses SHAP (SHapley Additive exPlanations) to explain why the XGBoost
model classifies network flows as Attack or Benign.  Outputs include:

- SHAP Summary Plot (beeswarm): per-feature impact direction and magnitude
- SHAP Bar Plot: global feature importance ranking
- SHAP Waterfall Plots: individual Attack / Benign prediction explanations
- SHAP Feature Importance CSV: machine-readable ranking
- Comparative analysis: SHAP importance vs. RF importance (from Exp 1)

Requirements: ``pip install shap``

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    get_classifiers,
    train_and_evaluate,
    compute_metrics,
    print_metrics,
    print_summary_table,
    shap_explain_model,
    export_multi_section_report,
    Pipeline,
    VectorAssembler,
    StandardScaler,
)

# ==============================================================================
# INITIALIZATION
# ==============================================================================
spark = create_spark_session("IDS_Exp5_SHAP_Explainability")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("=" * 70)
print("  EXPERIMENT 5: SHAP EXPLAINABILITY (XAI FOR IDS)")
print("=" * 70)
print(f"  Total Features: {len(feature_cols)}")

# Create output directory
base_output = "/Users/thainguyenvu/Desktop/Thesis_IDS/exp5_results"
os.makedirs(base_output, exist_ok=True)

# ==============================================================================
# STEP 1: TRAIN XGBOOST MODEL (Full Features)
# ==============================================================================
print("\n--- Step 1: Training XGBoost on All Features ---")

assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep",
)
scaler = StandardScaler(
    inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
)

# Get XGBoost classifier
classifiers = get_classifiers(
    features_col="features_scaled",
    label_col="label_binary",
    num_features=len(feature_cols),
)

# Train XGBoost only
xgb_clf = classifiers["XGBoost"]
xgb_pipeline = Pipeline(stages=[assembler, scaler, xgb_clf])

xgb_model, xgb_preds, xgb_metrics = train_and_evaluate(
    xgb_pipeline, train_df, test_df, title="XGBoost (for SHAP)"
)

print_metrics(xgb_metrics, title="XGBoost Performance")

results = {"XGBoost": xgb_metrics}

# ==============================================================================
# STEP 2: SHAP ANALYSIS
# ==============================================================================
print("\n--- Step 2: SHAP Explainability Analysis ---")

shap_output_dir = os.path.join(base_output, "shap_analysis")
shap_plots = shap_explain_model(
    spark_model=xgb_model,
    test_df=test_df,
    feature_cols=feature_cols,
    output_dir=shap_output_dir,
    sample_size=2000,  # 2000 samples for robust SHAP estimation
)

# ==============================================================================
# STEP 3: COMPARE SHAP vs RF FEATURE IMPORTANCE
# ==============================================================================
print("\n--- Step 3: SHAP vs RF Feature Importance Comparison ---")

# Load RF Feature Importance from Exp 1 (if available)
rf_importance_path = "/Users/thainguyenvu/Desktop/Thesis_IDS/feature_importance.csv"
shap_importance_path = os.path.join(shap_output_dir, "shap_feature_importance.csv")

if os.path.exists(rf_importance_path) and os.path.exists(shap_importance_path):
    rf_imp = pd.read_csv(rf_importance_path)
    shap_imp = pd.read_csv(shap_importance_path)
    
    # Normalize both to [0, 1] for comparison
    rf_imp_col = "importance" if "importance" in rf_imp.columns else rf_imp.columns[1]
    rf_imp["RF_Importance_Norm"] = rf_imp[rf_imp_col] / rf_imp[rf_imp_col].max()
    shap_imp["SHAP_Importance_Norm"] = shap_imp["Mean_SHAP_Value"] / shap_imp["Mean_SHAP_Value"].max()
    
    # Rename for merge
    rf_col_name = "feature" if "feature" in rf_imp.columns else rf_imp.columns[0]
    rf_imp = rf_imp.rename(columns={rf_col_name: "Feature"})
    
    # Merge on feature name
    comparison = pd.merge(
        shap_imp[["Feature", "SHAP_Importance_Norm"]],
        rf_imp[["Feature", "RF_Importance_Norm"]],
        on="Feature",
        how="outer"
    ).fillna(0)
    
    # Sort by SHAP importance
    comparison = comparison.sort_values("SHAP_Importance_Norm", ascending=False)
    
    # Save comparison
    comparison_path = os.path.join(base_output, "shap_vs_rf_importance.csv")
    comparison.to_csv(comparison_path, index=False)
    
    # Plot side-by-side comparison (Top-15)
    top_n = 15
    top_features = comparison.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(top_n)
    width = 0.35
    
    bars1 = ax.barh(x + width/2, top_features["SHAP_Importance_Norm"], width, 
                     label="SHAP Importance", color="#2196F3", alpha=0.85)
    bars2 = ax.barh(x - width/2, top_features["RF_Importance_Norm"], width, 
                     label="RF Feature Importance", color="#FF9800", alpha=0.85)
    
    ax.set_xlabel("Normalized Importance", fontsize=12)
    ax.set_title("SHAP vs Random Forest Feature Importance (Top-15)", fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(top_features["Feature"], fontsize=10)
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    
    comparison_plot_path = os.path.join(base_output, "shap_vs_rf_comparison.png")
    plt.savefig(comparison_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  [INFO] Comparison plot saved: {comparison_plot_path}")
    print(f"  [INFO] Comparison CSV saved: {comparison_path}")
    
    # Print Top-10 comparison
    print(f"\n  {'─' * 60}")
    print(f"  Top-10 Feature Importance: SHAP vs RF")
    print(f"  {'─' * 60}")
    print(f"  {'Rank':<6} {'Feature':<30} {'SHAP':>10} {'RF':>10}")
    print(f"  {'─' * 60}")
    for rank, (_, row) in enumerate(comparison.head(10).iterrows(), 1):
        print(f"  {rank:<6} {row['Feature']:<30} {row['SHAP_Importance_Norm']:>10.4f} {row['RF_Importance_Norm']:>10.4f}")
    
    shap_plots["comparison_plot"] = comparison_plot_path
else:
    print("  [WARN] RF Feature Importance file not found. Skipping comparison.")
    print(f"  Expected path: {rf_importance_path}")
    print("  Run exp1_rf_feature_importance.py first to generate it.")

# ==============================================================================
# VISUALIZATION & REPORTING
# ==============================================================================
print_summary_table(results, title="RESULTS: XGBOOST (SHAP ANALYSIS)")

# Collect all chart paths
all_chart_paths = [path for key, path in shap_plots.items() 
                   if path.endswith('.png')]

report_sections = [{
    "section_title": "XGBoost Performance for SHAP Analysis",
    "results": results,
    "chart_paths": all_chart_paths,
}]

export_multi_section_report(
    report_sections,
    title="IDS Thesis - Experiment 5: SHAP Explainability (XAI)",
    output_path=os.path.join(base_output, "exp5_report.html"),
)

print("\n[INFO] Experiment 5 completed!")
print(f"[INFO] Results exported to: {base_output}")
print(f"[INFO] SHAP analysis saved to: {os.path.join(base_output, 'shap_analysis')}")
spark.stop()
print("[INFO] Spark Session closed.")
