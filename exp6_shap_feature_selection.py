#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reporting import export_multi_section_report

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    train_and_evaluate,
    get_classifiers,
    run_all_classifiers,
    ensemble_voting,
    plot_comparison,
    plot_training_time,
    plot_prediction_time,
    plot_model_size,
    plot_confusion_matrices,
    plot_roc_curves,
    print_summary_table,
    Pipeline,
    VectorAssembler,
    StandardScaler,
)

spark = create_spark_session("IDS_Exp6_SHAP_Feature_Selection")
_, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("=" * 70)
print("  EXPERIMENT 6: FEATURE SELECTION - SHAP IMPORTANCE")
print("=" * 70)

base_output = os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "exp6_results_shap")
os.makedirs(base_output, exist_ok=True)


print("\n--- Step 1: Train XGBoost & Compute SHAP Feature Ranking ---")

assembler_all = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep",
)
scaler_all = StandardScaler(
    inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
)

classifiers = get_classifiers(
    features_col="features_scaled",
    label_col="label_binary",
    num_features=len(feature_cols),
)

xgb_clf = classifiers["XGBoost"]
pipeline_xgb = Pipeline(stages=[assembler_all, scaler_all, xgb_clf])

model_xgb, _, _ = train_and_evaluate(
    pipeline_xgb, train_df, test_df,
    title="XGBoost Baseline (extract SHAP Importance)"
)

print("\n  Computing SHAP values on training sample (to avoid data leakage)...")
import shap

sample_size = 2000
select_cols = feature_cols + ["label_binary"]
sample_df = train_df.select(select_cols).limit(sample_size)
pdf = sample_df.toPandas()
X_sample = pdf[feature_cols].values

xgb_model = None
for stage in model_xgb.stages:
    if "XGB" in type(stage).__name__:
        xgb_model = stage
        break

booster = xgb_model.get_booster()
print(f"  Booster extracted: {booster.num_boosted_rounds()} rounds")

explainer = shap.TreeExplainer(booster)
shap_values = explainer(X_sample)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "shap_importance": mean_abs_shap
}).sort_values("shap_importance", ascending=False).reset_index(drop=True)

print("\nTop 20 most important features (SHAP):")
print(shap_importance_df.head(20).to_string(index=True))

plt.figure(figsize=(12, 8))
top_n = 30
top_features = shap_importance_df.head(top_n)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
plt.barh(range(top_n), top_features["shap_importance"].values[::-1], color=colors[::-1])
plt.yticks(range(top_n), top_features["feature"].values[::-1])
plt.xlabel("Mean |SHAP Value|")
plt.title(f"Top {top_n} Feature Importance (SHAP - XGBoost)")
plt.tight_layout()

feat_imp_path = os.path.join(base_output, f"shap_feature_importance_top{top_n}.png")
plt.savefig(feat_imp_path, dpi=150)
plt.close()

shap_csv_path = os.path.join(base_output, "shap_feature_importance.csv")
shap_importance_df.to_csv(shap_csv_path, index=False)
print(f"[INFO] Saved: {shap_csv_path}")


all_results = {}
top_k_values = [20, 30, 40]
report_sections = []

report_sections.append({
    "section_title": "SHAP Feature Importance Analysis",
    "results": {},
    "chart_paths": [feat_imp_path]
})

for top_k in top_k_values:
    print(f"\n\n{'=' * 70}")
    print(f"  EVALUATING ALL ALGORITHMS WITH SHAP TOP-{top_k} FEATURES")
    print(f"{'=' * 70}")

    selected_features = shap_importance_df.head(top_k)["feature"].tolist()

    assembler_sel = VectorAssembler(
        inputCols=selected_features, outputCol="features_raw", handleInvalid="keep",
    )
    scaler_sel = StandardScaler(
        inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
    )

    results, trained_models = run_all_classifiers(
        assembler=assembler_sel,
        scaler=scaler_sel,
        train_df=train_df,
        test_df=test_df,
        features_col="features_scaled",
        num_features=top_k,
    )

    ens_metrics = ensemble_voting(trained_models, test_df, results=results)
    if ens_metrics:
        results["Ensemble Voting"] = ens_metrics

    all_results[f"SHAP-Top-{top_k}"] = results

    print_summary_table(results, title=f"RESULTS: SHAP TOP-{top_k} FEATURES")

    k_dir = os.path.join(base_output, f"shap_top{top_k}")
    os.makedirs(k_dir, exist_ok=True)

    plot_comparison(
        results,
        title=f"Exp 6: SHAP Top-{top_k} Features Comparison",
        save_path=os.path.join(k_dir, "comparison.png"),
        show=False,
    )
    plot_training_time(
        results,
        title=f"Exp 6: SHAP Top-{top_k} Training Time",
        save_path=os.path.join(k_dir, "train_time.png"),
        show=False,
    )
    plot_prediction_time(
        results,
        title=f"Exp 6: SHAP Top-{top_k} Prediction Time",
        save_path=os.path.join(k_dir, "pred_time.png"),
        show=False,
    )
    plot_model_size(
        results,
        title=f"Exp 6: SHAP Top-{top_k} Model Size",
        save_path=os.path.join(k_dir, "model_size.png"),
        show=False,
    )
    plot_confusion_matrices(
        results,
        title=f"Exp 6: SHAP Top-{top_k} Confusion Matrices",
        save_path=os.path.join(k_dir, "confusion_matrices.png"),
        show=False,
    )
    plot_roc_curves(
        trained_models, test_df,
        title=f"Exp 6: SHAP Top-{top_k} ROC Curves",
        save_path=os.path.join(k_dir, "roc_curves.png"),
        show=False,
    )

    report_sections.append({
        "section_title": f"SHAP Top-{top_k} Features Evaluation",
        "results": results,
        "chart_paths": [
            os.path.join(k_dir, "comparison.png"),
            os.path.join(k_dir, "train_time.png"),
            os.path.join(k_dir, "pred_time.png"),
            os.path.join(k_dir, "model_size.png"),
            os.path.join(k_dir, "confusion_matrices.png"),
            os.path.join(k_dir, "roc_curves.png"),
        ]
    })


print("\n\n" + "=" * 70)
print("  STEP 3: SHAP vs RF FEATURE SELECTION COMPARISON")
print("=" * 70)

rf_importance_path = os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "feature_importance.csv")

if os.path.exists(rf_importance_path):
    rf_imp = pd.read_csv(rf_importance_path)
    
    print(f"\n  {'─' * 65}")
    print(f"  {'Top-K':<10} {'Method':<20} {'Best Model':<25} {'F1-Score':>10}")
    print(f"  {'─' * 65}")
    
    for top_k in top_k_values:
        shap_results = all_results[f"SHAP-Top-{top_k}"]
        shap_best = max(shap_results, key=lambda k: shap_results[k].get("f1", 0))
        shap_f1 = shap_results[shap_best]["f1"]
        print(f"  {top_k:<10} {'SHAP':<20} {shap_best:<25} {shap_f1:>10.6f}")
        
        rf_result_note = "(Run Exp 1 to compare)"
        print(f"  {'':<10} {'RF Importance':<20} {rf_result_note:<25}")
        print(f"  {'─' * 65}")
    
    print(f"\n  Feature Overlap Analysis (SHAP vs RF):")
    print(f"  {'─' * 50}")
    for top_k in top_k_values:
        shap_top = set(shap_importance_df.head(top_k)["feature"].tolist())
        rf_top = set(rf_imp.head(top_k)["feature"].tolist())
        overlap = shap_top & rf_top
        only_shap = shap_top - rf_top
        only_rf = rf_top - shap_top
        print(f"\n  Top-{top_k}:")
        print(f"    Common features:    {len(overlap)}/{top_k} ({len(overlap)/top_k*100:.0f}%)")
        print(f"    SHAP-only features: {len(only_shap)}")
        print(f"    RF-only features:   {len(only_rf)}")
        if only_shap:
            print(f"    SHAP unique: {', '.join(list(only_shap)[:5])}{'...' if len(only_shap)>5 else ''}")
else:
    print("  RF Feature Importance not found. Run exp1 first for comparison.")


print("\n\n" + "=" * 70)
print("  EXPERIMENT 6 SUMMARY (SHAP FEATURE SELECTION)")
print("=" * 70)

for top_k_label, results in all_results.items():
    best_name = max(results, key=lambda k: results[k].get("f1", 0))
    best_f1 = results[best_name]["f1"]
    print(f"\n  {top_k_label}: Best = {best_name} (F1={best_f1:.6f})")

export_multi_section_report(
    report_sections,
    title="IDS Thesis - Experiment 6: SHAP Feature Selection Analysis",
    output_path=os.path.join(base_output, "exp6_report.html"),
)

print(f"\n[INFO] Experiment 6 completed!")
print(f"[INFO] Results exported to: {base_output}")
spark.stop()
print("[INFO] Spark Session closed.")
