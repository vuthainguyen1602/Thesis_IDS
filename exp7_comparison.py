#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reporting import export_multi_section_report

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    run_all_classifiers,
    ensemble_voting,
    compute_metrics,
    get_classifiers,
    train_and_evaluate,
    summarize_metric_runs,
    permutation_pvalue,
    plot_comparison,
    plot_training_time,
    print_summary_table,
    Pipeline,
    VectorAssembler,
    StandardScaler,
    PCA,
)


BASE_DIR = os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "exp7_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RF_IMPORTANCE_CSV = os.path.join(BASE_DIR, "feature_importance.csv")
SHAP_IMPORTANCE_CSV = os.path.join(BASE_DIR, "exp6_results_shap", "shap_feature_importance.csv")

METHODS = {
    "Baseline (All Features)": {"type": "all"},
    "RF Top-30": {"type": "feature_selection", "csv": RF_IMPORTANCE_CSV, "top_k": 30, "col": "feature"},
    "SHAP Top-30": {"type": "feature_selection", "csv": SHAP_IMPORTANCE_CSV, "top_k": 30, "col": "feature"},
    "PCA k=40": {"type": "pca", "k": 40},
}

STAT_SEEDS = [42, 52, 62]


def _method_slug(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")


def _build_method_transform(config, feature_cols):
    extra_stages = []
    if config["type"] == "all":
        selected_features = feature_cols
        assembler = VectorAssembler(inputCols=selected_features, outputCol="features_raw", handleInvalid="keep")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
        return assembler, scaler, extra_stages, "features_scaled", len(selected_features)

    if config["type"] == "feature_selection":
        csv_path = config["csv"]
        top_k = config["top_k"]
        col_name = config["col"]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")
        importance_df = pd.read_csv(csv_path)
        selected_features = importance_df.head(top_k)[col_name].tolist()
        assembler = VectorAssembler(inputCols=selected_features, outputCol="features_raw", handleInvalid="keep")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
        return assembler, scaler, extra_stages, "features_scaled", top_k

    if config["type"] == "pca":
        k = config["k"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
        pca = PCA(k=k, inputCol="features_scaled", outputCol="pca_features")
        extra_stages = [pca]
        return assembler, scaler, extra_stages, "pca_features", k

    raise ValueError(f"Unknown method type: {config['type']}")


def _train_single_named_model(method_cfg, model_name, feature_cols, train_df, test_df, seed: int):
    assembler, scaler, extra_stages, features_col, num_features = _build_method_transform(method_cfg, feature_cols)
    class_counts = train_df.groupBy("label_binary").count().collect()
    count_map = {row["label_binary"]: row["count"] for row in class_counts}
    benign = count_map.get(0, 0)
    attack = count_map.get(1, 0)
    scale_pos_weight = float(benign) / float(attack) if attack > 0 else 1.0

    clf_dict = get_classifiers(
        features_col=features_col, label_col="label_binary",
        num_features=num_features, scale_pos_weight=scale_pos_weight, seed=seed,
    )
    if model_name not in clf_dict:
        raise ValueError(f"Model '{model_name}' not available for method.")
    pipeline = Pipeline(stages=[assembler, scaler] + extra_stages + [clf_dict[model_name]])
    model, preds, metrics = train_and_evaluate(
        pipeline, train_df, test_df, title=f"{model_name} | Seed={seed}"
    )
    return model, preds, metrics


def _build_drift_windows(df):
    if "timestamp" in df.columns:
        ts_df = (
            df.withColumn("_ts_num", F.unix_timestamp("timestamp").cast("double"))
            .filter(F.col("_ts_num").isNotNull())
        )
        if ts_df.count() > 0:
            q60, q80 = ts_df.approxQuantile("_ts_num", [0.6, 0.8], 0.01)
            train_early = ts_df.filter(F.col("_ts_num") <= q60).drop("_ts_num")
            test_mid = ts_df.filter((F.col("_ts_num") > q60) & (F.col("_ts_num") <= q80)).drop("_ts_num")
            test_late = ts_df.filter(F.col("_ts_num") > q80).drop("_ts_num")
            return train_early, test_mid, test_late, "timestamp"

    train_early, test_mid, test_late = df.randomSplit([0.6, 0.2, 0.2], seed=2026)
    return train_early, test_mid, test_late, "random_split_fallback"


if __name__ == "__main__":

    spark = create_spark_session("IDS_Exp7_Comparison")
    df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

    print("\n")
    print("=" * 70)
    print("  EXPERIMENT 7: CROSS-EXPERIMENT COMPARISON")
    print("=" * 70)
    print(f"  Total original features: {len(feature_cols)}")
    print(f"  Methods to compare: {len(METHODS)}")


    all_method_results = {}
    all_method_models = {}
    report_sections = []

    for method_name, config in METHODS.items():
        print(f"\n\n{'=' * 70}")
        print(f"  METHOD: {method_name}")
        print(f"{'=' * 70}")

        extra_stages = []

        if config["type"] == "all":
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

        results, trained_models = run_all_classifiers(
            assembler=assembler,
            scaler=scaler,
            train_df=train_df,
            test_df=test_df,
            features_col=features_col,
            num_features=num_features,
            extra_stages=extra_stages,
        )

        ens_metrics = ensemble_voting(trained_models, test_df, results=results)
        if ens_metrics:
            results["Ensemble Voting"] = ens_metrics

        all_method_results[method_name] = results
        print_summary_table(results, title=f"RESULTS: {method_name}")

        all_method_models[method_name] = trained_models

        method_dir = os.path.join(OUTPUT_DIR, _method_slug(method_name))
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

        report_sections.append({
            "section_title": method_name,
            "results": results,
            "chart_paths": [
                os.path.join(method_dir, "comparison.png"),
                os.path.join(method_dir, "train_time.png"),
            ]
        })


    print(f"\n\n{'=' * 70}")
    print("  STEP 2: CROSS-METHOD COMPARISON")
    print(f"{'=' * 70}")

    all_models = []
    for method_results in all_method_results.values():
        for model_name in method_results:
            if model_name not in all_models:
                all_models.append(model_name)

    method_names = list(all_method_results.keys())
    n_methods = len(method_names)
    n_models = len(all_models)

    f1_matrix = np.zeros((n_methods, n_models))
    for i, method in enumerate(method_names):
        for j, model in enumerate(all_models):
            f1_matrix[i, j] = all_method_results[method].get(model, {}).get("f1", 0)

    fig, ax = plt.subplots(figsize=(18, 10))
    x = np.arange(n_models)
    bar_width = 0.8 / n_methods
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, method in enumerate(method_names):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, f1_matrix[i], bar_width,
                      label=method, color=colors[i % len(colors)], alpha=0.85)
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


    fig, ax = plt.subplots(figsize=(16, 6))
    import seaborn as sns

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


    print(f"\n\n{'=' * 70}")
    print("  EXPERIMENT 7: OVERALL SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'─' * 70}")
    print(f"  {'Method':<25} {'Best Model':<35} {'F1-Score':>10}")
    print(f"  {'─' * 70}")
    for method, info in best_per_method.items():
        print(f"  {method:<25} {info['best_model']:<35} {info['best_f1']:>10.6f}")
    print(f"  {'─' * 70}")

    overall_best_method = max(best_per_method, key=lambda m: best_per_method[m]["best_f1"])
    overall_info = best_per_method[overall_best_method]
    print(f"\n  ★ OVERALL BEST: {overall_best_method}")
    print(f"    Model: {overall_info['best_model']}")
    print(f"    F1:    {overall_info['best_f1']:.6f}")

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

    print(f"\n\n{'=' * 70}")
    print("  STEP 3: ROBUSTNESS TRACK (ADDITIONAL HOLDOUT)")
    print(f"{'=' * 70}")

    robust_data_dir = os.environ.get("IDS_ROBUST_DATA_DIR")
    if robust_data_dir:
        robust_test_path = os.path.join(robust_data_dir, "test_data.parquet")
        if os.path.exists(robust_test_path):
            robust_test_df = spark.read.parquet(robust_test_path)
            print(f"[INFO] Loaded external robustness test set: {robust_test_path}")
        else:
            print(f"[WARN] IDS_ROBUST_DATA_DIR set, but test_data.parquet not found. Using fallback split.")
            _, robust_test_df = df.randomSplit([0.8, 0.2], seed=2026)
    else:
        print("[INFO] IDS_ROBUST_DATA_DIR not set. Using in-domain alternative split for robustness.")
        _, robust_test_df = df.randomSplit([0.8, 0.2], seed=2026)

    robustness_rows = []
    for method_name, info in best_per_method.items():
        best_model_name = info["best_model"]
        best_model = all_method_models[method_name][best_model_name]
        robust_preds = best_model.transform(robust_test_df)
        robust_metrics = compute_metrics(robust_preds)
        robustness_rows.append({
            "Method": method_name,
            "Best_Model": best_model_name,
            "robust_f1": robust_metrics.get("f1"),
            "robust_auc_pr": robust_metrics.get("auc_pr"),
            "robust_auc_roc": robust_metrics.get("auc_roc"),
        })
    robustness_df = pd.DataFrame(robustness_rows).sort_values("robust_f1", ascending=False)
    robustness_csv = os.path.join(OUTPUT_DIR, "robustness_holdout_summary.csv")
    robustness_df.to_csv(robustness_csv, index=False)
    print(f"[INFO] Saved: {robustness_csv}")

    plt.figure(figsize=(12, 6))
    plt.barh(robustness_df["Method"], robustness_df["robust_f1"], color="#1f77b4", alpha=0.85)
    plt.xlabel("F1 on Robustness Holdout")
    plt.title("Robustness Track: Best Model per Method")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    robustness_plot = os.path.join(OUTPUT_DIR, "robustness_holdout_f1.png")
    plt.savefig(robustness_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {robustness_plot}")

    print(f"\n\n{'=' * 70}")
    print("  STEP 4: DRIFT SIMULATION TRACK")
    print(f"{'=' * 70}")

    drift_train, drift_mid, drift_late, drift_mode = _build_drift_windows(df)
    print(f"[INFO] Drift split mode: {drift_mode}")
    print(f"  Train(Early): {drift_train.count():,} | Mid: {drift_mid.count():,} | Late: {drift_late.count():,}")

    drift_method = overall_best_method
    drift_model_name = overall_info["best_model"]
    drift_cfg = METHODS[drift_method]

    model_early, pred_mid, metrics_mid = _train_single_named_model(
        drift_cfg, drift_model_name, feature_cols, drift_train, drift_mid, seed=42
    )
    pred_late_no_update = model_early.transform(drift_late)
    metrics_late_no_update = compute_metrics(pred_late_no_update)

    drift_retrain_df = drift_train.unionByName(drift_mid)
    _, pred_late_retrained, metrics_late_retrained = _train_single_named_model(
        drift_cfg, drift_model_name, feature_cols, drift_retrain_df, drift_late, seed=42
    )

    drift_rows = [
        {"Scenario": "Early->Mid", "f1": metrics_mid.get("f1"), "auc_pr": metrics_mid.get("auc_pr")},
        {"Scenario": "Early->Late (No Update)", "f1": metrics_late_no_update.get("f1"), "auc_pr": metrics_late_no_update.get("auc_pr")},
        {"Scenario": "Early+Mid->Late (Retrained)", "f1": metrics_late_retrained.get("f1"), "auc_pr": metrics_late_retrained.get("auc_pr")},
    ]
    drift_df = pd.DataFrame(drift_rows)
    drift_csv = os.path.join(OUTPUT_DIR, "drift_simulation_summary.csv")
    drift_df.to_csv(drift_csv, index=False)
    print(f"[INFO] Saved: {drift_csv}")

    plt.figure(figsize=(10, 5))
    plt.bar(drift_df["Scenario"], drift_df["f1"], color=["#42A5F5", "#EF5350", "#66BB6A"])
    plt.ylabel("F1-Score")
    plt.title(f"Drift Simulation ({drift_method} / {drift_model_name})")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    drift_plot = os.path.join(OUTPUT_DIR, "drift_simulation_f1.png")
    plt.savefig(drift_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {drift_plot}")

    print(f"\n\n{'=' * 70}")
    print("  STEP 5: STATISTICAL VALIDITY TRACK (MULTI-SEED)")
    print(f"{'=' * 70}")

    sorted_methods = sorted(best_per_method.keys(), key=lambda m: best_per_method[m]["best_f1"], reverse=True)
    top_methods = sorted_methods[:2]
    stats_records = []
    method_seed_scores = {}

    for method_name in top_methods:
        model_name = best_per_method[method_name]["best_model"]
        cfg = METHODS[method_name]
        seed_metrics = []
        seed_f1_scores = []
        for s in STAT_SEEDS:
            _, _, m = _train_single_named_model(cfg, model_name, feature_cols, train_df, test_df, seed=s)
            seed_metrics.append(m)
            seed_f1_scores.append(float(m.get("f1", 0.0)))
        agg = summarize_metric_runs(seed_metrics, metric_keys=["f1", "auc_pr", "accuracy"])
        method_seed_scores[method_name] = seed_f1_scores
        stats_records.append({
            "Method": method_name,
            "Model": model_name,
            "Seeds": ",".join([str(s) for s in STAT_SEEDS]),
            "f1_scores": ",".join([f"{v:.6f}" for v in seed_f1_scores]),
            "f1_mean": agg.get("f1_mean"),
            "f1_std": agg.get("f1_std"),
            "f1_ci95_low": agg.get("f1_ci95_low"),
            "f1_ci95_high": agg.get("f1_ci95_high"),
        })

    pvalue = 1.0
    if len(top_methods) == 2:
        pvalue = permutation_pvalue(
            method_seed_scores[top_methods[0]],
            method_seed_scores[top_methods[1]],
            n_permutations=2000,
            seed=42,
        )
        print(f"[INFO] Permutation p-value ({top_methods[0]} vs {top_methods[1]}): {pvalue:.6f}")

    stats_df = pd.DataFrame(stats_records)
    stats_df["pvalue_vs_other_top_method"] = pvalue
    stats_csv = os.path.join(OUTPUT_DIR, "statistical_validity_multiseed.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[INFO] Saved: {stats_csv}")

    plt.figure(figsize=(10, 5))
    for method_name in top_methods:
        plt.plot(STAT_SEEDS, method_seed_scores[method_name], marker="o", label=method_name)
    plt.xlabel("Seed")
    plt.ylabel("F1-Score")
    plt.title("Multi-Seed Stability (Top Methods)")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    stats_plot = os.path.join(OUTPUT_DIR, "multiseed_stability.png")
    plt.savefig(stats_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {stats_plot}")


    report_sections.append({
        "section_title": "Cross-Method Comparison",
        "results": {},
        "chart_paths": [cross_f1_path, best_f1_path, heatmap_path],
    })

    report_sections.append({
        "section_title": "Robustness Track",
        "results": {},
        "chart_paths": [robustness_plot],
    })
    report_sections.append({
        "section_title": "Drift Simulation Track",
        "results": {},
        "chart_paths": [drift_plot],
    })
    report_sections.append({
        "section_title": "Statistical Validity Track",
        "results": {},
        "chart_paths": [stats_plot],
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
