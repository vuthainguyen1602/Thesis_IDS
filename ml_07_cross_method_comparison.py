#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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
    add_class_weights,
    train_and_evaluate,
    summarize_metric_runs,
    permutation_pvalue,
    paired_cohens_d,
    plot_comparison,
    plot_training_time,
    print_summary_table,
    ml_results_dir,
    shared_results_path,
    ML02_DIR,
    ML06_DIR,
    ML07_DIR,
    Pipeline,
    VectorAssembler,
    StandardScaler,
    PCA,
)


OUTPUT_DIR = ml_results_dir(ML07_DIR)
RF_IMPORTANCE_CSV = ml_results_dir(ML02_DIR, "feature_importance.csv")
SHAP_IMPORTANCE_CSV = ml_results_dir(ML06_DIR, "shap_feature_importance.csv")

METHODS = {
    "Baseline (All Features)": {"type": "all"},
    "RF Top-30": {"type": "feature_selection", "csv": RF_IMPORTANCE_CSV, "top_k": 30, "col": "feature"},
    "SHAP Top-30": {"type": "feature_selection", "csv": SHAP_IMPORTANCE_CSV, "top_k": 30, "col": "feature"},
    "PCA k=40": {"type": "pca", "k": 40},
}

# Statistical-validity design: variance comes from repeated DATA resampling
# (paired train/test splits), not from classifier-init seeds on a fixed split.
# The paired test is TWO-SIDED (uses |mean diff|), so the smallest reachable
# p-value is 2/2^N. N=6 -> 2/64 = 0.031 is the minimum that can still cross 0.05;
# a fixed single split / 3 seeds could never go below ~0.125. Override with IDS_STAT_SPLITS
# (use >=6 to keep significance reachable; lower N is fine if you lead with the
# CI + effect size instead of a hard p<0.05 claim). Classifier seed is fixed so
# the only varying factor is the data split.
STAT_N_SPLITS = int(os.environ.get("IDS_STAT_SPLITS", "6"))
STAT_SPLIT_SEED_BASE = 1000
STAT_CLF_SEED = 42

_META_ENSEMBLE_MARKERS = ("Bagging", "Ensemble", "Voting")


def _is_meta_ensemble(model_name: str) -> bool:
    return any(marker in model_name for marker in _META_ENSEMBLE_MARKERS)


def _pick_best_model(results: dict, single_classifiers_only: bool = False):
    """Select the best model per method on VALIDATION F1 when available.

    Selecting on test F1 would make every downstream number (stat track,
    robustness, per-class) conditional on a test-set peek — exactly the
    selection leakage the paper claims to close. ``val_f1`` is produced by
    run_all_classifiers(val_df=...); the test-F1 fallback exists only for
    legacy CSVs and warns loudly.
    """
    candidates = results
    if single_classifiers_only:
        filtered = {k: v for k, v in results.items() if not _is_meta_ensemble(k)}
        if filtered:
            candidates = filtered
    has_val = any(v.get("val_f1") is not None for v in candidates.values())
    rank_key = "val_f1" if has_val else "f1"
    if not has_val:
        print("  [WARN] val_f1 missing from results — falling back to TEST F1 for "
              "model selection. This reintroduces selection leakage; regenerate "
              "results with a validation split before publishing.")
    best_model = max(candidates, key=lambda k: candidates[k].get(rank_key) or 0)
    # Return the ranking score (validation-based when available) so downstream
    # method ordering is also leakage-free.
    return best_model, candidates[best_model].get(rank_key) or 0


def _build_best_per_method(all_method_results: dict) -> dict:
    best_per_method = {}
    for method, results in all_method_results.items():
        best_model, best_f1 = _pick_best_model(results, single_classifiers_only=True)
        overall_best, overall_f1 = _pick_best_model(results, single_classifiers_only=False)
        if _is_meta_ensemble(overall_best) and overall_best != best_model:
            print(
                f"  [INFO] {method}: ensemble '{overall_best}' (F1={overall_f1:.6f}) "
                f"excluded from downstream tracks; using '{best_model}' (F1={best_f1:.6f})"
            )
        best_per_method[method] = {"best_model": best_model, "best_f1": best_f1}
    return best_per_method


_METRIC_KEYS = [
    "accuracy", "precision", "recall", "f1", "val_f1", "auc_roc", "auc_pr",
    "training_time", "prediction_time", "model_size_mb",
]


def _load_results_from_summary_csv(csv_path: str) -> dict:
    summary_df = pd.read_csv(csv_path)
    all_method_results = {}
    for _, row in summary_df.iterrows():
        method = row["Method"]
        model = row["Model"]
        all_method_results.setdefault(method, {})
        all_method_results[method][model] = {
            key: (row[key] if key in row and pd.notna(row[key]) else None)
            for key in _METRIC_KEYS
        }
    return all_method_results


def _trainable_model_name(method_name: str, model_name: str, all_method_results: dict) -> str:
    if not _is_meta_ensemble(model_name):
        return model_name
    if method_name not in all_method_results:
        raise ValueError(f"Unknown method '{method_name}' when resolving trainable model.")
    resolved, f1 = _pick_best_model(all_method_results[method_name], single_classifiers_only=True)
    print(
        f"  [WARN] Cannot retrain ensemble '{model_name}'; "
        f"using single classifier '{resolved}' (F1={f1:.6f})"
    )
    return resolved


def _run_statistical_validity_track(
    best_per_method, all_method_results, feature_cols, df, output_dir,
):
    print(f"\n\n{'=' * 70}")
    print(f"  STEP 5: STATISTICAL VALIDITY TRACK ({STAT_N_SPLITS} REPEATED SPLITS, PAIRED)")
    print(f"{'=' * 70}")
    print("  Variance source: repeated stratified train/test resampling of the")
    print("  full dataset. Both top methods are trained on the SAME split each")
    print("  iteration (paired design); classifier seed is fixed.")
    if STAT_N_SPLITS < 6:
        print(f"  [WARN] STAT_N_SPLITS={STAT_N_SPLITS} is low — the two-sided paired "
              f"permutation test can only reach p>=2/2^{STAT_N_SPLITS}="
              f"{2.0/(2**STAT_N_SPLITS):.3f}. Use >=6 for p<0.05, or rely on the CI.")

    # Both the model per method and the pair of methods entering the test are
    # ranked on VALIDATION F1 (best_per_method comes from _pick_best_model,
    # which prefers val_f1) — the test set plays no role in these choices, so
    # the p-value below is not conditioned on a test-set peek.
    sorted_methods = sorted(best_per_method.keys(), key=lambda m: best_per_method[m]["best_f1"], reverse=True)
    top_methods = sorted_methods[:2]

    # Resolve the (validation-selected, fixed before testing) model per method once.
    method_model = {
        m: _trainable_model_name(m, best_per_method[m]["best_model"], all_method_results)
        for m in top_methods
    }

    # method -> list of F1 across splits (index-aligned => paired across methods).
    method_split_f1 = {m: [] for m in top_methods}
    method_split_metrics = {m: [] for m in top_methods}

    df = df.cache()  # scanned once per split below; materialise the union once
    df.count()

    for i in range(STAT_N_SPLITS):
        split_seed = STAT_SPLIT_SEED_BASE + i
        # Fresh stratified-ish resample of the whole labelled dataset. Using the
        # full df (not the fixed train_df) is what injects data-sampling
        # variance; randomSplit on a large set preserves the class ratio well.
        train_s, test_s = df.randomSplit([0.8, 0.2], seed=split_seed)
        train_s = train_s.cache(); test_s = test_s.cache()
        train_s.count(); test_s.count()
        print(f"\n  ── Split {i + 1}/{STAT_N_SPLITS} (seed={split_seed}) ──")
        for method_name in top_methods:
            cfg = METHODS[method_name]
            _, _, m = _train_single_named_model(
                cfg, method_model[method_name], feature_cols, train_s, test_s, seed=STAT_CLF_SEED,
            )
            method_split_f1[method_name].append(float(m.get("f1", 0.0)))
            method_split_metrics[method_name].append(m)
        train_s.unpersist(); test_s.unpersist()

    df.unpersist()
    stats_records = []
    for method_name in top_methods:
        # Nadeau–Bengio correction: the N resamples share overlapping training
        # data, so the naive t-CI is optimistically narrow. rho = n_test/n_train
        # = 0.2/0.8 = 0.25 for the 80/20 resampling used here. The paper reports
        # the NB interval (f1_ci95_nb_*); the naive one is kept for comparison.
        agg = summarize_metric_runs(
            method_split_metrics[method_name],
            metric_keys=["f1", "auc_pr", "accuracy"],
            nb_test_train_ratio=0.25,
        )
        stats_records.append({
            "Method": method_name,
            "Model": method_model[method_name],
            "N_splits": STAT_N_SPLITS,
            "f1_scores": ",".join(f"{v:.6f}" for v in method_split_f1[method_name]),
            "f1_mean": agg.get("f1_mean"),
            "f1_std": agg.get("f1_std"),
            "f1_ci95_low": agg.get("f1_ci95_low"),
            "f1_ci95_high": agg.get("f1_ci95_high"),
            "f1_ci95_nb_low": agg.get("f1_ci95_nb_low"),
            "f1_ci95_nb_high": agg.get("f1_ci95_nb_high"),
        })

    pvalue = 1.0
    effect_size = 0.0
    if len(top_methods) == 2:
        # Paired permutation test on the per-split F1 differences. With N=6 the
        # 2^6=64 sign patterns are enumerated exactly inside permutation_pvalue.
        pvalue = permutation_pvalue(
            method_split_f1[top_methods[0]],
            method_split_f1[top_methods[1]],
            n_permutations=2000,
            seed=42,
        )
        effect_size = paired_cohens_d(
            method_split_f1[top_methods[0]],
            method_split_f1[top_methods[1]],
        )
        min_p = 2.0 / (2 ** STAT_N_SPLITS)  # two-sided sign-permutation floor
        print(f"\n[INFO] Paired permutation p-value ({top_methods[0]} vs "
              f"{top_methods[1]}): {pvalue:.6f}  (exact enumeration; two-sided "
              f"min achievable = {min_p:.4f})")
        print(f"[INFO] Paired Cohen's d (effect size): {effect_size:.3f} — report "
              "alongside the p-value; with N=6 a non-significant p is weak "
              "evidence of equivalence on its own.")

    stats_df = pd.DataFrame(stats_records)
    stats_df["pvalue_vs_other_top_method"] = pvalue
    stats_df["paired_cohens_d"] = effect_size
    stats_csv = os.path.join(output_dir, "statistical_validity_multiseed.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[INFO] Saved: {stats_csv}")

    x = list(range(1, STAT_N_SPLITS + 1))
    plt.figure(figsize=(10, 5))
    for method_name in top_methods:
        plt.plot(x, method_split_f1[method_name], marker="o", label=method_name)
    plt.xlabel("Resample split #")
    plt.ylabel("F1-Score")
    plt.title(f"Per-Split F1 across {STAT_N_SPLITS} Resamples (Top Methods)")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    stats_plot = os.path.join(output_dir, "multiseed_stability.png")
    plt.savefig(stats_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {stats_plot}")
    return stats_plot, stats_csv


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
    # Balance class weights uniformly (weightCol-aware models read this column).
    train_df = add_class_weights(train_df, scale_pos_weight)
    pipeline = Pipeline(stages=[assembler, scaler] + extra_stages + [clf_dict[model_name]])
    model, preds, metrics = train_and_evaluate(
        pipeline, train_df, test_df, title=f"{model_name} | Seed={seed}"
    )
    return model, preds, metrics


def _build_drift_windows(df):
    # The union DataFrame arrives uncached; cache here since the drift
    # simulation scans it several times.
    df = df.cache()
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

    print("\n" + "!" * 70)
    print("  [WARN] No usable 'timestamp' column found in the dataset.")
    print("  Drift windows fall back to a RANDOM split, which contains NO")
    print("  temporal/concept drift. Results from this track are NOT a valid")
    print("  concept-drift evaluation and must NOT be reported as such.")
    print("  Provide timestamped data to enable a real temporal drift test.")
    print("!" * 70 + "\n")
    train_early, test_mid, test_late = df.randomSplit([0.6, 0.2, 0.2], seed=2026)
    return train_early, test_mid, test_late, "random_split_fallback"


if __name__ == "__main__":

    spark = create_spark_session("IDS_Exp7_Comparison")
    df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

    start_step = int(os.environ.get("IDS_EXP7_START_STEP") or "1")

    if start_step >= 5:
        summary_csv = os.path.join(OUTPUT_DIR, "cross_method_summary.csv")
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(
                f"IDS_EXP7_START_STEP=5 requires prior Step 1–2 output: {summary_csv}\n"
                "Run full ml_07 first, or lower IDS_EXP7_START_STEP."
            )
        print("\n")
        print("=" * 70)
        print("  EXPERIMENT 7: RESUME FROM STEP 5 (MULTI-SEED STATS)")
        print("=" * 70)
        print(f"[INFO] Loaded: {summary_csv}")
        all_method_results = _load_results_from_summary_csv(summary_csv)
        best_per_method = _build_best_per_method(all_method_results)
        for method, info in best_per_method.items():
            print(f"  {method:<25} -> {info['best_model']} (F1={info['best_f1']:.6f})")

        stats_plot, _ = _run_statistical_validity_track(
            best_per_method, all_method_results, feature_cols, df, OUTPUT_DIR,
        )
        export_multi_section_report(
            [{
                "section_title": "Statistical Validity Track",
                "results": {},
                "chart_paths": [stats_plot],
            }],
            title="IDS Thesis - Experiment 7: Statistical Validity (Resume)",
            output_path=os.path.join(OUTPUT_DIR, "report_step5.html"),
        )
        print(f"\n[INFO] Step 5 completed!")
        print(f"[INFO] Results exported to: {OUTPUT_DIR}")
        spark.stop()
        print("[INFO] Spark Session closed.")
        raise SystemExit(0)

    print("\n")
    print("=" * 70)
    print("  EXPERIMENT 7: CROSS-EXPERIMENT COMPARISON")
    print("=" * 70)
    print(f"  Total original features: {len(feature_cols)}")
    print(f"  Methods to compare: {len(METHODS)}")

    # Validation holdout carved from TRAIN (disjoint from TEST). Base models are
    # trained on train_core and the ensemble members are ranked by their F1 on
    # val_df, so the test set never participates in model selection — closing the
    # selection-leakage loop where members were previously chosen by test F1.
    train_core, val_df = train_df.randomSplit([0.8, 0.2], seed=42)
    train_core = train_core.cache()
    val_df = val_df.cache()
    print(f"  Selection holdout: base models train on train_core, ensemble members "
          f"ranked on val (both carved from TRAIN); test set untouched.")

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
            train_df=train_core,
            test_df=test_df,
            features_col=features_col,
            num_features=num_features,
            extra_stages=extra_stages,
            val_df=val_df,
        )

        # ensemble_voting reads val_f1 from `results` (set above) to rank its
        # members on validation, not test.
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


    best_per_method = _build_best_per_method(all_method_results)

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
            for key in _METRIC_KEYS:
                row[key] = metrics.get(key, None)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "cross_method_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")

    # ── Consolidated train vs. predict time for the BEST model per method ─────
    # Single cross-method timing figure for the papers (FAIR fig:timing), in
    # addition to the per-method train_time.png charts above. Times are measured
    # on the Spark executors, i.e. on the Jetson cluster nodes themselves.
    try:
        import numpy as _np
        # Exclude meta-ensemble rows (their training_time is recorded as 0.0 and
        # the downstream tracks use single classifiers); pick the best single
        # classifier per method so the timing figure matches the reported model.
        _single_df = summary_df[~summary_df["Model"].apply(_is_meta_ensemble)]
        if _single_df.empty:
            _single_df = summary_df
        # Match the leakage-free selection rule: rank on validation F1 when
        # available so the timing figure shows the same models the paper reports.
        _rank_col = ("val_f1" if ("val_f1" in _single_df.columns
                                  and _single_df["val_f1"].notna().any()) else "f1")
        best_idx = _single_df.groupby("Method")[_rank_col].idxmax()
        best = _single_df.loc[best_idx].copy()
        _order = ["Baseline", "RF Importance", "SHAP", "PCA"]
        best["__o"] = best["Method"].apply(
            lambda m: _order.index(m) if m in _order else 99)
        best = best.sort_values(["__o", "Method"])
        _labels = [f"{m}\n({mod})" for m, mod in zip(best["Method"], best["Model"])]
        _x = _np.arange(len(best))
        _w = 0.38
        figt, axt = plt.subplots(figsize=(9, 5))
        _b1 = axt.bar(_x - _w / 2, best["training_time"].astype(float), _w,
                      label="Training time (s)", color="#1f77b4")
        _b2 = axt.bar(_x + _w / 2, best["prediction_time"].astype(float), _w,
                      label="Prediction time (s)", color="#ff7f0e")
        axt.set_xticks(_x)
        axt.set_xticklabels(_labels, fontsize=8)
        axt.set_ylabel("Time (s)")
        axt.set_title("Best-model training vs. prediction time per method "
                      "(Jetson Spark cluster)")
        axt.legend()
        axt.bar_label(_b1, fmt="%.1f", fontsize=8, padding=2)
        axt.bar_label(_b2, fmt="%.2f", fontsize=8, padding=2)
        figt.tight_layout()
        _tp = os.path.join(OUTPUT_DIR, "train_predict_time.png")
        plt.savefig(_tp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved: {_tp}")
    except Exception as _e:
        print(f"[WARN] train/predict-time plot skipped: {_e}")


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
    config_path = shared_results_path("best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"[INFO] Saved Best Config for Exp 2: {config_path}")

    print(f"\n\n{'=' * 70}")
    print("  STEP 3: ROBUSTNESS TRACK (ADDITIONAL HOLDOUT)")
    print(f"{'=' * 70}")

    robust_data_dir = os.environ.get("IDS_ROBUST_DATA_DIR")
    robust_is_external = False
    if robust_data_dir:
        robust_test_path = os.path.join(robust_data_dir, "test_data.parquet")
        if os.path.exists(robust_test_path):
            robust_test_df = spark.read.parquet(robust_test_path)
            robust_is_external = True
            print(f"[INFO] Loaded external robustness test set: {robust_test_path}")
        else:
            print(f"[WARN] IDS_ROBUST_DATA_DIR set, but test_data.parquet not found. Falling back.")
            robust_test_df = None
    else:
        robust_test_df = None

    if robust_test_df is None:
        # IMPORTANT: must NOT resplit `df` (= train ∪ test) here — that would put
        # rows the models were trained on into the "holdout", leaking and
        # inflating robustness F1. Draw a subsample of test_df instead, which is
        # disjoint from train by construction. This is only an in-domain proxy
        # (same distribution), NOT a true distribution-shift robustness test.
        robust_test_df = test_df.sample(withReplacement=False, fraction=0.7, seed=2026)
        print("[WARN] No external robustness set (IDS_ROBUST_DATA_DIR). Using a "
              "leakage-free in-domain subsample of the test set as a PROXY — "
              "this is not a true robustness/distribution-shift evaluation.")

    robust_test_df = robust_test_df.cache()
    print(f"  Robustness set: {robust_test_df.count():,} rows "
          f"({'external' if robust_is_external else 'in-domain proxy'})")

    robustness_rows = []
    for method_name, info in best_per_method.items():
        best_model_name = info["best_model"]
        best_model = all_method_models[method_name][best_model_name]
        robust_preds = best_model.transform(robust_test_df).cache()
        robust_preds.count()
        robust_metrics = compute_metrics(robust_preds)
        robust_preds.unpersist()
        robustness_rows.append({
            "Method": method_name,
            "Best_Model": best_model_name,
            "robust_f1": robust_metrics.get("f1"),
            "robust_auc_pr": robust_metrics.get("auc_pr"),
            "robust_auc_roc": robust_metrics.get("auc_roc"),
        })
    robust_test_df.unpersist()
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
    drift_is_valid = drift_mode == "timestamp"
    if not drift_is_valid:
        print("[WARN] Drift track ran in random-split fallback — NOT a valid "
              "concept-drift result (flagged as drift_valid=False in the CSV).")
    print(f"  Train(Early): {drift_train.count():,} | Mid: {drift_mid.count():,} | Late: {drift_late.count():,}")

    drift_method = overall_best_method
    drift_model_name = _trainable_model_name(
        drift_method, overall_info["best_model"], all_method_results,
    )
    drift_cfg = METHODS[drift_method]

    model_early, pred_mid, metrics_mid = _train_single_named_model(
        drift_cfg, drift_model_name, feature_cols, drift_train, drift_mid, seed=42
    )
    # Cache once so compute_metrics' three passes don't re-run inference 3x.
    pred_late_no_update = model_early.transform(drift_late).cache()
    pred_late_no_update.count()
    metrics_late_no_update = compute_metrics(pred_late_no_update)
    pred_late_no_update.unpersist()

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
    drift_df["drift_mode"] = drift_mode
    drift_df["drift_valid"] = drift_is_valid
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

    stats_plot, _ = _run_statistical_validity_track(
        best_per_method, all_method_results, feature_cols, df, OUTPUT_DIR,
    )

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
        output_path=os.path.join(OUTPUT_DIR, "report.html"),
    )

    print(f"\n[INFO] Experiment 7 completed!")
    print(f"[INFO] Results exported to: {OUTPUT_DIR}")
    spark.stop()
    print("[INFO] Spark Session closed.")
