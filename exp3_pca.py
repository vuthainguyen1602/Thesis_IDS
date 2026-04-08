#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    export_results_to_html,
    export_multi_section_report,
    Pipeline,
    VectorAssembler,
    StandardScaler,
    PCA,
)

spark = create_spark_session("IDS_Exp3_PCA")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("=" * 70)
print("  EXPERIMENT 3: DIMENSIONALITY REDUCTION USING PCA")
print("=" * 70)

base_output = os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "exp3_results")
os.makedirs(base_output, exist_ok=True)

print("\n--- Step 1: Explained Variance Analysis ---")

assembler_pca = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep",
)
scaler_pca = StandardScaler(
    inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
)

k_max = min(len(feature_cols), 60)
pca_full = PCA(k=k_max, inputCol="features_scaled", outputCol="pca_features_full")
pipeline_full = Pipeline(stages=[assembler_pca, scaler_pca, pca_full])
model_full = pipeline_full.fit(train_df)

pca_model_full = model_full.stages[2]
explained_var_full = pca_model_full.explainedVariance.toArray()
cumulative_var_full = np.cumsum(explained_var_full)

for threshold in [0.90, 0.95, 0.99]:
    k_needed = np.searchsorted(cumulative_var_full, threshold) + 1
    print(f"  Need k={k_needed} components to explain {threshold*100:.0f}% variance")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(1, len(explained_var_full) + 1), explained_var_full, color="steelblue", alpha=0.7)
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance Ratio")
ax1.set_title("Individual Explained Variance")

ax2.plot(range(1, len(cumulative_var_full) + 1), cumulative_var_full, "bo-", markersize=3)
ax2.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
ax2.axhline(y=0.99, color="g", linestyle="--", label="99% threshold")
ax2.set_xlabel("Number of Components")
ax2.set_ylabel("Cumulative Variance")
ax2.set_title("Cumulative Explained Variance")
ax2.legend()

plt.suptitle("PCA - Explained Variance Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
feat_imp_path = os.path.join(base_output, "exp3_explained_variance.png")
plt.savefig(feat_imp_path, dpi=150)
plt.close()
print(f"[INFO] Saved: {feat_imp_path}")


all_exp3_results = {}
report_sections = []

report_sections.append({
    "section_title": "Explained Variance Analysis",
    "results": {},
    "chart_paths": [feat_imp_path]
})

for k in [15, 25, 35]:
    print(f"\n\n{'=' * 70}")
    print(f"  EVALUATING ALL ALGORITHMS WITH PCA k={k}")
    print(f"{'=' * 70}")
    print(f"  Cumulative variance with k={k}: {cumulative_var_full[k-1]*100:.2f}%")

    pca = PCA(k=k, inputCol="features_scaled", outputCol="pca_features")

    results, trained_models = run_all_classifiers(
        assembler=assembler_pca, scaler=scaler_pca,
        train_df=train_df, test_df=test_df,
        features_col="pca_features", num_features=k,
        extra_stages=[pca],
    )

    ens_metrics = ensemble_voting(trained_models, test_df, results=results)
    if ens_metrics:
        results["Ensemble Voting"] = ens_metrics

    all_exp3_results[k] = results
    print_summary_table(results, title=f"RESULTS: PCA k={k}")

    k_dir = os.path.join(base_output, f"k{k}")
    os.makedirs(k_dir, exist_ok=True)

    plot_comparison(
        results,
        title=f"Experiment 3: Algorithm Comparison (PCA k={k})",
        save_path=os.path.join(k_dir, "comparison.png"),
        show=False,
    )
    plot_training_time(
        results,
        title=f"Experiment 3: Training Time (PCA k={k})",
        save_path=os.path.join(k_dir, "train_time.png"),
        show=False,
    )
    plot_prediction_time(
        results,
        title=f"Experiment 3: Prediction Time (PCA k={k})",
        save_path=os.path.join(k_dir, "pred_time.png"),
        show=False,
    )
    plot_model_size(
        results,
        title=f"Experiment 3: Model Size (PCA k={k})",
        save_path=os.path.join(k_dir, "model_size.png"),
        show=False,
    )
    plot_confusion_matrices(
        results,
        title=f"Experiment 3: Confusion Matrices (PCA k={k})",
        save_path=os.path.join(k_dir, "confusion_matrices.png"),
        show=False,
    )
    plot_roc_curves(
        trained_models, test_df,
        title=f"Experiment 3: ROC Curves (PCA k={k})",
        save_path=os.path.join(k_dir, "roc_curves.png"),
        show=False,
    )

    report_sections.append({
        "section_title": f"PCA Dimensionality Reduction (k={k})",
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


print(f"\n\n{'=' * 70}")
print("  EXPERIMENT 3 SUMMARY")
print(f"{'=' * 70}")

all_algos = set()
for k in all_exp3_results:
    all_algos.update(all_exp3_results[k].keys())

summary_data = {}
for algo in sorted(all_algos):
    row = {}
    for k in [15, 25, 35]:
        if algo in all_exp3_results.get(k, {}):
            row[f"k={k} F1"] = all_exp3_results[k][algo].get("f1", 0)
            row[f"k={k} AUC"] = all_exp3_results[k][algo].get("auc_roc", None)
    summary_data[algo] = row

summary_df = pd.DataFrame(summary_data).T
print("\nF1-Score & AUC-ROC for each algorithm across PCA k values:")
print(summary_df.to_string())

best_config, best_f1 = None, 0
for k, results in all_exp3_results.items():
    for algo, m in results.items():
        if m.get("f1", 0) > best_f1:
            best_f1 = m["f1"]
            best_config = (k, algo)

if best_config:
    print(f"\n★ Best config: PCA k={best_config[0]}, {best_config[1]} → F1={best_f1:.6f}")

export_multi_section_report(
    report_sections, 
    title="IDS Thesis - Experiment 3: PCA Dimensionality Reduction",
    output_path=os.path.join(base_output, "exp3_report.html")
)

print("\n[INFO] Experiment 3 completed!")
spark.stop()
print("[INFO] Spark Session closed.")
