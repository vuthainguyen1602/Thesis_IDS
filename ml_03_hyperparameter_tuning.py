#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import pandas as pd
from reporting import export_multi_section_report

from shared_utils import (
    HAS_XGBOOST,
    MultilayerPerceptronClassifier,
    SparkXGBClassifier,
    create_spark_session,
    load_and_prepare_data,
    compute_metrics,
    plot_comparison,
    plot_training_time,
    plot_roc_curves,
    print_summary_table,
    ensemble_voting,
    train_hybrid_bagging,
    Pipeline,
    VectorAssembler,
    StandardScaler,
    RandomForestClassifier,
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    BinaryClassificationEvaluator,
    ParamGridBuilder,
    CrossValidator,
    PCA,
    get_model_size,
    _get_best_params,
    ml_results_dir,
    shared_results_path,
    ML03_DIR,
    ML02_DIR,
)


def _build_param_grid(estimator, spec, fast_mode):
    builder = ParamGridBuilder()
    for param, values in spec:
        builder = builder.addGrid(param, values(fast_mode) if callable(values) else values)
    return builder.build()


def _format_best_params(params, keys):
    parts = [f"{k}={params.get(k)}" for k in keys]
    return ", ".join(parts)


def run_grid_search_tuning(
    *,
    step_id,
    display_name,
    section_title,
    base_stages,
    estimator,
    grid_spec,
    best_log_keys,
    train_cv_df,
    test_df,
    evaluator,
    cv_folds,
    parallelism,
    fast_mode,
):
    """Fit CrossValidator, evaluate on test set, return metrics + best params."""
    print(f"\n{'━' * 70}\n  {step_id}. Grid Search + CV: {section_title.upper()}\n{'━' * 70}")
    pipeline = Pipeline(stages=base_stages + [estimator])
    param_grid = _build_param_grid(estimator, grid_spec, fast_mode)

    start = time.time()
    cv_model = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=cv_folds,
        parallelism=parallelism,
        seed=42,
    ).fit(train_cv_df)
    train_time = time.time() - start

    best_params = _get_best_params(cv_model, param_grid)
    print(f"[BEST] {display_name}: {_format_best_params(best_params, best_log_keys)}")

    # transform() is lazy; cache + force one action so pred_time measures real
    # inference and compute_metrics' three passes don't re-run inference 3x.
    predictions = cv_model.bestModel.transform(test_df).cache()
    start_pred = time.time()
    predictions.count()
    pred_time = time.time() - start_pred

    metrics = compute_metrics(predictions)
    metrics["training_time"] = train_time
    metrics["prediction_time"] = pred_time
    metrics["model_size_mb"] = get_model_size(cv_model.bestModel)
    predictions.unpersist()

    result_key = f"{display_name} (Tuned)"
    return result_key, metrics, cv_model.bestModel, best_params


def _tuning_jobs(features_col, num_features, fast_mode, run_gbt, run_extended):
    """Return tuning job specs: (step_id, enabled, skip_msg, parallelism, job_kwargs)."""
    jobs = []

    def add(step_id, enabled, skip_msg, parallelism, display_name, section_title, estimator, grid_spec, best_log_keys, store_key=None):
        jobs.append({
            "step_id": step_id,
            "enabled": enabled,
            "skip_msg": skip_msg,
            "parallelism": parallelism,
            "display_name": display_name,
            "section_title": section_title,
            "estimator": estimator,
            "grid_spec": grid_spec,
            "best_log_keys": best_log_keys,
            "store_key": store_key or display_name.lower().replace(" ", "_"),
        })

    rf = RandomForestClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
    add(
        "2a", True, None, 4, "Random Forest", "Random Forest", rf,
        [
            (rf.numTrees, lambda fast: [200] if fast else [200, 300]),
            (rf.maxDepth, [15, 20]),
            (rf.minInstancesPerNode, lambda fast: [1] if fast else [1, 5]),
        ],
        ["numTrees", "maxDepth", "minInstancesPerNode"],
        store_key="rf",
    )

    dt = DecisionTreeClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
    add(
        "2b", True, None, 4, "Decision Tree", "Decision Tree", dt,
        [
            (dt.maxDepth, [15, 20]),
            (dt.impurity, lambda fast: ["gini"] if fast else ["gini", "entropy"]),
            (dt.minInstancesPerNode, [1, 5]),
        ],
        ["maxDepth", "impurity"],
    )

    gbt = GBTClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
    add(
        "2c", run_gbt,
        "Skipping GBT tuning (fast mode). Set IDS_EXP2_FULL=1 or IDS_EXP2_GBT=1 to enable.",
        2, "GBT", "GBT", gbt,
        [
            (gbt.maxIter, lambda fast: [100] if fast else [100, 150]),
            (gbt.maxDepth, [6, 8]),
            (gbt.stepSize, lambda fast: [0.1] if fast else [0.05, 0.1]),
        ],
        ["maxIter", "maxDepth"],
    )

    lr = LogisticRegression(featuresCol=features_col, labelCol="label_binary", family="binomial")
    add(
        "2d", True, None, 4, "Logistic Regression", "Logistic Regression", lr,
        [
            (lr.regParam, [0.01, 0.1]),
            (lr.elasticNetParam, lambda fast: [0.5] if fast else [0.5, 0.8]),
        ],
        ["regParam", "elasticNetParam"],
    )

    if run_extended and HAS_XGBOOST:
        xgb = SparkXGBClassifier(features_col=features_col, label_col="label_binary", num_workers=4)
        add(
            "2e", True, None, 2, "XGBoost", "XGBoost", xgb,
            [
                (xgb.n_estimators, [100, 300]),
                (xgb.max_depth, [5, 8]),
                (xgb.learning_rate, [0.05, 0.1]),
            ],
            ["max_depth", "learning_rate"],
            store_key="xgb",
        )

    # LightGBM tuning excluded (not deployable on Jetson Orin Nano Super ARM64).

    if run_extended:
        layers_opts = [[num_features, 64, 32, 2], [num_features, 32, 16, 2]]
        mlp = MultilayerPerceptronClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
        add(
            "2g", True, None, 4, "MLP", "MLP", mlp,
            [
                (mlp.layers, layers_opts),
                (mlp.maxIter, [50, 100]),
                (mlp.stepSize, [0.01, 0.05]),
            ],
            ["layers", "maxIter"],
        )

    return jobs


def _build_hybrid_bagging_pipelines(base_stages, features_col, tuned_params, run_extended):
    rf_params = tuned_params.get("rf", {})
    rf_tuned = RandomForestClassifier(
        featuresCol=features_col,
        labelCol="label_binary",
        numTrees=int(rf_params.get("numTrees", 200)),
        maxDepth=int(rf_params.get("maxDepth", 15)),
        minInstancesPerNode=int(rf_params.get("minInstancesPerNode", 1)),
        seed=42,
    )
    pipeline_dist = [(Pipeline(stages=base_stages + [rf_tuned]), 3)]

    if run_extended and HAS_XGBOOST and tuned_params.get("xgb"):
        xgb_params = tuned_params["xgb"]
        xgb_tuned = SparkXGBClassifier(
            features_col=features_col,
            label_col="label_binary",
            max_depth=int(xgb_params.get("max_depth", 8)),
            learning_rate=float(xgb_params.get("learning_rate", 0.05)),
            num_workers=4,
        )
        pipeline_dist.append((Pipeline(stages=base_stages + [xgb_tuned]), 2))

    # LightGBM tuned-ensemble branch excluded (not deployable on Jetson Orin Nano Super).

    return pipeline_dist


spark = create_spark_session("IDS_Exp2_GridSearch_CV")
_, train_df, test_df, feature_cols = load_and_prepare_data(spark)

# ── Tuning profile (default: FAST — set IDS_EXP2_FULL=1 for full run) ──
FULL_MODE = os.environ.get("IDS_EXP2_FULL", "0").strip().lower() in ("1", "true", "yes")
FAST_MODE = not FULL_MODE
if os.environ.get("IDS_EXP2_FAST") is not None:
    FAST_MODE = os.environ.get("IDS_EXP2_FAST", "1").strip().lower() in ("1", "true", "yes")

CV_FOLDS = int(os.environ.get("IDS_CV_FOLDS", "3"))
CV_FRACTION = float(os.environ.get("IDS_CV_FRACTION", "0.15" if FAST_MODE else "1.0"))
RUN_EXTENDED = (
    FULL_MODE
    and os.environ.get("IDS_EXP2_EXTENDED", "0").strip().lower() in ("1", "true", "yes")
)
RUN_GBT = FULL_MODE or os.environ.get("IDS_EXP2_GBT", "0").strip().lower() in ("1", "true", "yes")

train_cv_df = train_df
if CV_FRACTION < 1.0:
    train_cv_df = train_df.sample(withReplacement=False, fraction=CV_FRACTION, seed=42).cache()
    print(f"[INFO] CV subset: {train_cv_df.count():,} rows ({CV_FRACTION:.0%} of train)")

print(
    f"[CONFIG] mode={'fast' if FAST_MODE else 'full'}, folds={CV_FOLDS}, "
    f"cv_sample={CV_FRACTION:.0%}, gbt={RUN_GBT}, extended={RUN_EXTENDED}"
)

print("\n")
print("=" * 70)
print("  EXPERIMENT 2: GRID SEARCH + CROSS-VALIDATION")
print("=" * 70)

print("\n--- Step 1: Loading Best Configuration from Exp 7 ---")

config_path = shared_results_path("best_config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        best_config = json.load(f)
    print(f"[INFO] Loaded best configuration: {best_config['method_name']}")
else:
    print("[WARN] best_config.json not found. Run Exp 7 first.")
    print("Falling back to RF Feature Selection (Top-40) as default.")
    best_config = {
        "method_name": "RF Top-40 (Fallback)",
        "config": {
            "type": "feature_selection",
            "csv": ml_results_dir(ML02_DIR, "feature_importance.csv"),
            "top_k": 40,
            "col": "feature",
        },
    }

method_cfg = best_config["config"]
extra_stages = []

if method_cfg["type"] == "feature_selection":
    csv_path = method_cfg["csv"]
    top_k = method_cfg["top_k"]
    col_name = method_cfg["col"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Feature selection file not found: {csv_path}\n"
            "Run ml_02_feature_selection_rf.py (or ml_06_feature_selection_shap.py) first."
        )
    importance_df = pd.read_csv(csv_path)
    selected_features = importance_df.head(top_k)[col_name].tolist()
    print(f"Using {len(selected_features)} features from {os.path.basename(csv_path)}")

    assembler_cv = VectorAssembler(inputCols=selected_features, outputCol="features_raw", handleInvalid="keep")
    scaler_cv = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
    features_col = "features_scaled"
    num_features = top_k

elif method_cfg["type"] == "pca":
    k = method_cfg["k"]
    print(f"Using PCA with k={k} components")
    assembler_cv = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
    scaler_cv = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
    pca = PCA(k=k, inputCol="features_scaled", outputCol="pca_features")
    extra_stages = [pca]
    features_col = "pca_features"
    num_features = k

else:
    print("Using all features")
    assembler_cv = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
    scaler_cv = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
    features_col = "features_scaled"
    num_features = len(feature_cols)

base_stages = [assembler_cv, scaler_cv] + extra_stages
evaluator_cv = BinaryClassificationEvaluator(
    labelCol="label_binary", rawPredictionCol="rawPrediction", metricName="areaUnderPR",
)

cv_results = {}
cv_models = {}
report_sections = []
tuned_params = {}
base_output = ml_results_dir(ML03_DIR)
os.makedirs(base_output, exist_ok=True)

for job in _tuning_jobs(features_col, num_features, FAST_MODE, RUN_GBT, RUN_EXTENDED):
    if not job["enabled"]:
        if job["skip_msg"]:
            print(f"\n[INFO] {job['skip_msg']}")
        continue

    result_key, metrics, best_model, best_params = run_grid_search_tuning(
        step_id=job["step_id"],
        display_name=job["display_name"],
        section_title=job["section_title"],
        base_stages=base_stages,
        estimator=job["estimator"],
        grid_spec=job["grid_spec"],
        best_log_keys=job["best_log_keys"],
        train_cv_df=train_cv_df,
        test_df=test_df,
        evaluator=evaluator_cv,
        cv_folds=CV_FOLDS,
        parallelism=job["parallelism"],
        fast_mode=FAST_MODE,
    )

    cv_results[result_key] = metrics
    cv_models[result_key] = best_model
    tuned_params[job["store_key"]] = best_params
    report_sections.append({
        "section_title": f"Tuning: {job['display_name']}",
        "results": {result_key: metrics},
    })

print(f"\n{'━' * 70}\n  2h. Ensemble Voting (Tuned Models)\n{'━' * 70}")
ens_metrics = ensemble_voting(cv_models, test_df, results=cv_results)
if ens_metrics:
    cv_results["Ensemble Voting (Tuned)"] = ens_metrics
    report_sections.append({
        "section_title": "Tuning: Ensemble",
        "results": {"Ensemble Voting (Tuned)": ens_metrics},
    })

print(f"\n{'━' * 70}\n  2i. Hybrid Bagging Ensemble (Tuned Models)\n{'━' * 70}")
pipeline_dist_tuned = _build_hybrid_bagging_pipelines(
    base_stages, features_col, tuned_params, RUN_EXTENDED,
)

start_bag_train = time.time()
bag_model_tuned = train_hybrid_bagging(pipeline_dist_tuned, train_df)
bag_train_time = time.time() - start_bag_train

if bag_model_tuned:
    start_pred = time.time()
    bag_preds_tuned = bag_model_tuned.transform(test_df)
    bag_preds_tuned.cache().count()
    bag_pred_time_tuned = time.time() - start_pred

    metrics_bag_tuned = compute_metrics(bag_preds_tuned)
    metrics_bag_tuned["training_time"] = bag_train_time
    metrics_bag_tuned["prediction_time"] = bag_pred_time_tuned
    metrics_bag_tuned["model_size_mb"] = sum(get_model_size(m) for m in bag_model_tuned.models)

    cv_results["Hybrid Bagging Ensemble (Tuned)"] = metrics_bag_tuned
    report_sections.append({
        "section_title": "Tuning: Hybrid Bagging",
        "results": {"Hybrid Bagging Ensemble (Tuned)": metrics_bag_tuned},
    })

print(f"\n\n{'=' * 70}\n  STEP 3: CONSOLIDATED TUNED RESULTS\n{'=' * 70}")

print_summary_table(cv_results, title=f"GRID SEARCH ON BEST CONFIG: {best_config['method_name']}")

plot_comparison(
    cv_results,
    title="ml_03: Tuned Models Comparison",
    save_path=os.path.join(base_output, "comparison.png"),
    show=False,
)
plot_training_time(
    cv_results,
    title="ml_03: Tuning Time",
    save_path=os.path.join(base_output, "train_time.png"),
    show=False,
)
plot_roc_curves(
    cv_models,
    test_df,
    title="ml_03: Tuned Models ROC Curves",
    save_path=os.path.join(base_output, "roc_curves.png"),
    show=False,
)

report_sections.append({
    "section_title": "Final Comparison (Tuned Models)",
    "results": cv_results,
    "chart_paths": [
        os.path.join(base_output, "comparison.png"),
        os.path.join(base_output, "roc_curves.png"),
    ],
})

export_multi_section_report(
    report_sections,
    title=f"ml_03: Grid Search on {best_config['method_name']}",
    output_path=os.path.join(base_output, "report.html"),
)

print("\n[INFO] Experiment 2 completed!")
spark.stop()
