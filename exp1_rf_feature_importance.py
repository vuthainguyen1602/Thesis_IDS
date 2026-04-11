#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from reporting import export_multi_section_report, export_results_to_html

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    train_and_evaluate,
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
    RandomForestClassifier,
)

spark = create_spark_session("IDS_Exp1_RF_Feature_Importance")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("=" * 70)
print("  EXPERIMENT 1: FEATURE SELECTION - RF FEATURE IMPORTANCE")
print("=" * 70)


print("\n--- Step 1: Train RF Baseline to extract Feature Importance ---")

assembler_all = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep",
)
scaler_all = StandardScaler(
    inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
)
rf_baseline = RandomForestClassifier(
    featuresCol="features_scaled", labelCol="label_binary",
    numTrees=100, maxDepth=10, seed=42,
)

pipeline_baseline = Pipeline(stages=[assembler_all, scaler_all, rf_baseline])
model_baseline, _, _ = train_and_evaluate(
    pipeline_baseline, train_df, test_df,
    title="RF Baseline (extract Feature Importance)"
)

rf_model = model_baseline.stages[-1]
importances = rf_model.featureImportances.toArray()

feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 20 most important features:")
print(feature_importance_df.head(20).to_string(index=True))

plt.figure(figsize=(12, 8))
top_n = 30
top_features = feature_importance_df.head(top_n)
plt.barh(range(top_n), top_features["importance"].values[::-1])
plt.yticks(range(top_n), top_features["feature"].values[::-1])
plt.xlabel("Importance")
plt.title(f"Top {top_n} Feature Importance (Random Forest)")
plt.tight_layout()

base_output = os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "exp1_results")
os.makedirs(base_output, exist_ok=True)

feat_imp_path = os.path.join(base_output, f"feature_importance_top{top_n}.png")
plt.savefig(feat_imp_path, dpi=150)
plt.close()

feature_importance_df.to_csv(
    os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "feature_importance.csv"), index=False
)
print("[INFO] Saved: feature_importance.csv")


all_exp1_results = {}
top_k_values = [20, 30, 40]
report_sections = []

report_sections.append({
    "section_title": "Feature Importance Analysis",
    "results": {},
    "chart_paths": [feat_imp_path]
})

for top_k in top_k_values:
    print(f"\n\n{'=' * 70}")
    print(f"  EVALUATING ALL ALGORITHMS WITH TOP-{top_k} FEATURES")
    print(f"{'=' * 70}")

    selected_features = feature_importance_df.head(top_k)["feature"].tolist()

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

    all_exp1_results[f"Top-{top_k}"] = results

    print_summary_table(results, title=f"RESULTS: TOP-{top_k} FEATURES")

    k_dir = os.path.join(base_output, f"top{top_k}")
    os.makedirs(k_dir, exist_ok=True)

    plot_comparison(
        results,
        title=f"Experiment 1: Top-{top_k} Features Comparison",
        save_path=os.path.join(k_dir, f"comparison.png"),
        show=False,
    )
    plot_training_time(
        results,
        title=f"Experiment 1: Top-{top_k} Training Time",
        save_path=os.path.join(k_dir, f"train_time.png"),
        show=False,
    )
    plot_prediction_time(
        results,
        title=f"Experiment 1: Top-{top_k} Prediction Time",
        save_path=os.path.join(k_dir, f"pred_time.png"),
        show=False,
    )
    plot_model_size(
        results,
        title=f"Experiment 1: Top-{top_k} Model Size",
        save_path=os.path.join(k_dir, f"model_size.png"),
        show=False,
    )
    plot_confusion_matrices(
        results,
        title=f"Experiment 1: Top-{top_k} Confusion Matrices",
        save_path=os.path.join(k_dir, f"confusion_matrices.png"),
        show=False,
    )
    plot_roc_curves(
        trained_models, test_df,
        title=f"Experiment 1: Top-{top_k} ROC Curves",
        save_path=os.path.join(k_dir, f"roc_curves.png"),
        show=False,
    )

    report_sections.append({
        "section_title": f"Top-{top_k} Features Evaluation",
        "results": results,
        "chart_paths": [
            os.path.join(k_dir, f"comparison.png"),
            os.path.join(k_dir, f"train_time.png"),
            os.path.join(k_dir, f"pred_time.png"),
            os.path.join(k_dir, f"model_size.png"),
            os.path.join(k_dir, f"confusion_matrices.png"),
            os.path.join(k_dir, f"roc_curves.png"),
        ]
    })


print("\n\n" + "=" * 70)
print("  EXPERIMENT 1 SUMMARY")
print("=" * 70)

for top_k, results in all_exp1_results.items():
    best_name = max(results, key=lambda k: results[k].get("f1", 0))
    best_f1 = results[best_name]["f1"]
    print(f"\n  {top_k}: Best = {best_name} (F1={best_f1:.6f})")

export_multi_section_report(
    report_sections, 
    title="IDS Thesis - Experiment 1: RF Feature Importance Analysis",
    output_path=os.path.join(base_output, "exp1_report.html")
)

print("\n[INFO] Experiment 1 completed!")
spark.stop()
print("[INFO] Spark Session closed.")
