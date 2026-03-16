#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 7 — Cross-Experiment Comparison.

Re-runs the best configuration from each prior experiment side-by-side:

1. Baseline — all features (Exp 0)
2. RF Top-30 — RF importance-based selection (Exp 1)
3. SHAP Top-30 — SHAP-based selection (Exp 6)
4. PCA k=35 — PCA dimensionality reduction (Exp 3)

Produces: grouped bar chart, heatmap, summary CSV, and an overall best
configuration JSON consumed by Experiment 2 (Grid Search).

Requirements:
  - ``feature_importance.csv`` (from Exp 1)
  - ``exp6_results_shap/shap_feature_importance.csv`` (from Exp 6)

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    run_all_classifiers,
    ensemble_voting,
    plot_comparison,
    plot_training_time,
    print_summary_table,
    export_multi_section_report,
    Pipeline,
    VectorAssembler,
    StandardScaler,
    PCA,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = "/Users/thainguyenvu/Desktop/Thesis_IDS"
OUTPUT_DIR = os.path.join(BASE_DIR, "exp7_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RF_IMPORTANCE_CSV = os.path.join(BASE_DIR, "feature_importance.csv")
SHAP_IMPORTANCE_CSV = os.path.join(BASE_DIR, "exp6_results_shap", "shap_feature_importance.csv")

# Methods to compare
METHODS = {
    "Baseline (All Features)": {"type": "all"},
    "RF Top-30": {"type": "feature_selection", "csv": RF_IMPORTANCE_CSV, "top_k": 30, "col": "feature"},
    "SHAP Top-30": {"type": "feature_selection", "csv": SHAP_IMPORTANCE_CSV, "top_k": 30, "col": "feature"},
    "PCA k=35": {"type": "pca", "k": 35},
}


# ==============================================================================
# INITIALIZATION
# ==============================================================================
if __name__ == "__main__":

    spark = create_spark_session("IDS_Exp7_Comparison")
    df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

    print("\n")
    print("=" * 70)
    print("  EXPERIMENT 7: CROSS-EXPERIMENT COMPARISON")
    print("=" * 70)
    print(f"  Total original features: {len(feature_cols)}")
    print(f"  Methods to compare: {len(METHODS)}")


    # ==========================================================================
    # STEP 1: RUN ALL METHODS
    # ==========================================================================
    all_method_results = {}   # {method_name: {model_name: metrics_dict}}
    report_sections = []

    for method_name, config in METHODS.items():
        print(f"\n\n{'=' * 70}")
        print(f"  METHOD: {method_name}")
        print(f"{'=' * 70}")

        extra_stages = []

        if config["type"] == "all":
            # Baseline: use all features
            selected_features = feature_cols
            assembler = VectorAssembler(
                inputCols=selected_features, outputCol="features_raw", handleInvalid="keep",
            )
            scaler = StandardScaler(
                inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
            )
            features_col = "features_scaled"
            num_features = len(selected_features)

        elif config["type"] == "feature_selection":
            # RF or SHAP Feature Selection
            csv_path = config["csv"]
            top_k = config["top_k"]
            col_name = config["col"]

            if not os.path.exists(csv_path):
                print(f"  [ERROR] {csv_path} not found. Run the corresponding experiment first.")
                continue

            importance_df = pd.read_csv(csv_path)
            selected_features = importance_df.head(top_k)[col_name].tolist()
            print(f"  Selected {len(selected_features)} features from {os.path.basename(csv_path)}")

            assembler = VectorAssembler(
                inputCols=selected_features, outputCol="features_raw", handleInvalid="keep",
            )
            scaler = StandardScaler(
                inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
            )
            features_col = "features_scaled"
            num_features = top_k

        elif config["type"] == "pca":
            # PCA Dimensionality Reduction
            k = config["k"]
            assembler = VectorAssembler(
                inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep",
            )
            scaler = StandardScaler(
                inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True,
            )
            pca = PCA(k=k, inputCol="features_scaled", outputCol="pca_features")
            extra_stages = [pca]
            features_col = "pca_features"
            num_features = k

        # --- Run all classifiers ---
        results, trained_models = run_all_classifiers(
            assembler=assembler,
            scaler=scaler,
            train_df=train_df,
            test_df=test_df,
            features_col=features_col,
            num_features=num_features,
            extra_stages=extra_stages,
        )

        # Ensemble Voting
        ens_metrics = ensemble_voting(trained_models, test_df, results=results)
        if ens_metrics:
            results["Ensemble Voting"] = ens_metrics

        all_method_results[method_name] = results
        print_summary_table(results, title=f"RESULTS: {method_name}")

        # Save per-method comparison chart
        method_dir = os.path.join(OUTPUT_DIR, method_name.replace(" ", "_").replace("(", "").replace(")", ""))
        os.makedirs(method_dir, exist_ok=True)

        plot_comparison(
            results,
            title=f"Exp 7: {method_name}",
            save_path=os.path.join(method_dir, "comparison.png"),
            show=False,
        )
        plot_training_time(
            results,
            title=f"Exp 7: {method_name} - Training Time",
            save_path=os.path.join(method_dir, "train_time.png"),
            show=False,
        )

        # Collect for report
        report_sections.append({
            "section_title": method_name,
            "results": results,
            "chart_paths": [
                os.path.join(method_dir, "comparison.png"),
                os.path.join(method_dir, "train_time.png"),
            ]
        })


    # ==========================================================================
    # STEP 2: CROSS-METHOD COMPARISON CHARTS
    # ==========================================================================
    print(f"\n\n{'=' * 70}")
    print("  STEP 2: CROSS-METHOD COMPARISON")
    print(f"{'=' * 70}")

    # --- 2a. Grouped Bar Chart: F1 per model across all methods ---
    # Get union of all model names (only standalone + ensemble, skip duplicates)
    all_models = []
    for method_results in all_method_results.values():
        for model_name in method_results:
            if model_name not in all_models:
                all_models.append(model_name)

    method_names = list(all_method_results.keys())
    n_methods = len(method_names)
    n_models = len(all_models)

    # Build F1 matrix
    f1_matrix = np.zeros((n_methods, n_models))
    for i, method in enumerate(method_names):
        for j, model in enumerate(all_models):
            f1_matrix[i, j] = all_method_results[method].get(model, {}).get("f1", 0)

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(18, 10))
    x = np.arange(n_models)
    bar_width = 0.8 / n_methods
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, method in enumerate(method_names):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, f1_matrix[i], bar_width,
                      label=method, color=colors[i % len(colors)], alpha=0.85)
        # Add value labels on top
        for bar, val in zip(bars, f1_matrix[i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("Cross-Method F1-Score Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    cross_f1_path = os.path.join(OUTPUT_DIR, "cross_method_f1_comparison.png")
    plt.savefig(cross_f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {cross_f1_path}")


    # --- 2b. Best F1 per Method (Summary) ---
    best_per_method = {}
    for method, results in all_method_results.items():
        best_model = max(results, key=lambda k: results[k].get("f1", 0))
        best_f1 = results[best_model]["f1"]
        best_per_method[method] = {"best_model": best_model, "best_f1": best_f1}

    fig, ax = plt.subplots(figsize=(12, 6))
    methods_list = list(best_per_method.keys())
    f1_values = [best_per_method[m]["best_f1"] for m in methods_list]
    model_labels = [best_per_method[m]["best_model"] for m in methods_list]
    bar_colors = [colors[i % len(colors)] for i in range(len(methods_list))]

    bars = ax.barh(methods_list, f1_values, color=bar_colors, alpha=0.85, height=0.5)
    for bar, val, model in zip(bars, f1_values, model_labels):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.6f} ({model})", va="center", fontsize=10)

    ax.set_xlabel("Best F1-Score", fontsize=12)
    ax.set_title("Best F1-Score per Method", fontsize=15, fontweight="bold")
    ax.set_xlim(min(f1_values) - 0.02 if min(f1_values) > 0.02 else 0, 1.005)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    best_f1_path = os.path.join(OUTPUT_DIR, "best_f1_per_method.png")
    plt.savefig(best_f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {best_f1_path}")


    # --- 2c. Cross-Method Summary Table ---
    summary_rows = []
    for method, results in all_method_results.items():
        for model, metrics in results.items():
            row = {"Method": method, "Model": model}
            for key in ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr",
                         "training_time", "prediction_time", "model_size_mb"]:
                row[key] = metrics.get(key, None)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "cross_method_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")


    # --- 2d. Heatmap: F1-Score Method × Model ---
    fig, ax = plt.subplots(figsize=(16, 6))
    import seaborn as sns

    # Create pivot table
    heatmap_data = pd.DataFrame(f1_matrix, index=method_names, columns=all_models)

    sns.heatmap(
        heatmap_data, annot=True, fmt=".4f", cmap="YlGn",
        linewidths=0.5, ax=ax, annot_kws={"fontsize": 8},
        vmin=heatmap_data.values[heatmap_data.values > 0].min() - 0.01,
        vmax=1.0,
    )
    ax.set_title("F1-Score Heatmap: Method × Model", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    heatmap_path = os.path.join(OUTPUT_DIR, "f1_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {heatmap_path}")


    # ==========================================================================
    # STEP 3: OVERALL SUMMARY
    # ==========================================================================
    print(f"\n\n{'=' * 70}")
    print("  EXPERIMENT 7: OVERALL SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'─' * 70}")
    print(f"  {'Method':<25} {'Best Model':<35} {'F1-Score':>10}")
    print(f"  {'─' * 70}")
    for method, info in best_per_method.items():
        print(f"  {method:<25} {info['best_model']:<35} {info['best_f1']:>10.6f}")
    print(f"  {'─' * 70}")

    # Find overall winner
    overall_best_method = max(best_per_method, key=lambda m: best_per_method[m]["best_f1"])
    overall_info = best_per_method[overall_best_method]
    print(f"\n  ★ OVERALL BEST: {overall_best_method}")
    print(f"    Model: {overall_info['best_model']}")
    print(f"    F1:    {overall_info['best_f1']:.6f}")

    # --- Step 3b: Save Best Config for Experiment 2 ---
    import json
    best_config = {
        "method_name": overall_best_method,
        "config": METHODS[overall_best_method],
        "best_model": overall_info['best_model'],
        "best_f1": overall_info['best_f1']
    }
    config_path = os.path.join(BASE_DIR, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"[INFO] Saved Best Config for Exp 2: {config_path}")


    # ==========================================================================
    # STEP 4: EXPORT HTML REPORT
    # ==========================================================================

    # Add cross-method comparison section
    report_sections.append({
        "section_title": "Cross-Method Comparison",
        "results": {},
        "chart_paths": [cross_f1_path, best_f1_path, heatmap_path],
    })

    export_multi_section_report(
        report_sections,
        title="IDS Thesis - Experiment 7: Cross-Experiment Comparison",
        output_path=os.path.join(OUTPUT_DIR, "exp7_report.html"),
    )

    print(f"\n[INFO] Experiment 7 completed!")
    print(f"[INFO] Results exported to: {OUTPUT_DIR}")
    spark.stop()
    print("[INFO] Spark Session closed.")
