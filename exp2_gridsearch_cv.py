#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 2 — Hyperparameter Tuning via Grid Search and Cross-Validation.

Loads the best feature-engineering configuration from Experiment 7, then
applies Spark MLlib's ``CrossValidator`` with exhaustive parameter grids
for selected classifiers (RF, DT, GBT, LR).  Tuned results are compared
against the default-parameter baselines.

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    compute_metrics,
    print_metrics,
    run_all_classifiers,
    ensemble_voting,
    plot_comparison,
    plot_training_time,
    plot_prediction_time,
    plot_model_size,
    plot_confusion_matrices,
    plot_roc_curves,
    print_summary_table,
    export_multi_section_report,
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
)

# ==============================================================================
# INITIALIZATION
# ==============================================================================
spark = create_spark_session("IDS_Exp2_GridSearch_CV")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("=" * 70)
print("  EXPERIMENT 2: GRID SEARCH + CROSS-VALIDATION")
print("=" * 70)

# ==============================================================================
# STEP 1: LOAD BEST CONFIG FROM EXP 7
# ==============================================================================
print("\n--- Step 1: Loading Best Configuration from Exp 7 ---")

config_path = "/Users/thainguyenvu/Desktop/Thesis_IDS/best_config.json"
best_config = None

if os.path.exists(config_path):
    with open(config_path, "r") as f:
        best_config = json.load(f)
    print(f"[INFO] Loaded best configuration: {best_config['method_name']}")
else:
    print("[WARN] best_config.json not found. Run Exp 7 first.")
    print("Falling back to RF Feature Selection (Top-40) as default.")
    # Default fallback
    best_config = {
        "method_name": "RF Top-40 (Fallback)",
        "config": {"type": "feature_selection", "csv": "/Users/thainguyenvu/Desktop/Thesis_IDS/feature_importance.csv", "top_k": 40, "col": "feature"}
    }

# Setup Pipeline Stages based on config
method_cfg = best_config["config"]
extra_stages = []
selected_features = feature_cols

if method_cfg["type"] == "feature_selection":
    csv_path = method_cfg["csv"]
    top_k = method_cfg["top_k"]
    col_name = method_cfg["col"]
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

else: # "all"
    print("Using all features")
    assembler_cv = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
    scaler_cv = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
    features_col = "features_scaled"
    num_features = len(feature_cols)

# Evaluate with PR-AUC for Grid Search
evaluator_cv = BinaryClassificationEvaluator(
    labelCol="label_binary", rawPredictionCol="rawPrediction", metricName="areaUnderPR",
)


# ==============================================================================
# STEP 2: HYPERPARAMETER OPTIMIZATION (GRID SEARCH + CV)
# ==============================================================================
cv_results = {}
report_sections = []
base_output = "/Users/thainguyenvu/Desktop/Thesis_IDS/exp2_results"
os.makedirs(base_output, exist_ok=True)

# 2a. Random Forest
print(f"\n{'━' * 70}\n  2a. Grid Search + CV: RANDOM FOREST\n{'━' * 70}")
rf = RandomForestClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
pipeline_rf = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [rf])

rf_grid = ParamGridBuilder().addGrid(rf.numTrees, [200, 300]).addGrid(rf.maxDepth, [15, 20]).build()
start = time.time()
cv_rf = CrossValidator(estimator=pipeline_rf, estimatorParamMaps=rf_grid, evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_rf_model = cv_rf.fit(train_df)
cv_rf_time = time.time() - start

start_pred = time.time()
predictions_rf = cv_rf_model.bestModel.transform(test_df)
rf_pred_time = time.time() - start_pred

metrics_rf = compute_metrics(predictions_rf)
metrics_rf["training_time"] = cv_rf_time
metrics_rf["prediction_time"] = rf_pred_time
metrics_rf["model_size_mb"] = get_model_size(cv_rf_model.bestModel)
cv_results["Random Forest (Tuned)"] = metrics_rf
report_sections.append({"section_title": "Tuning: Random Forest", "results": {"Random Forest (Tuned)": metrics_rf}})

# 2b. Decision Tree
print(f"\n{'━' * 70}\n  2b. Grid Search + CV: DECISION TREE\n{'━' * 70}")
dt = DecisionTreeClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
pipeline_dt = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [dt])

dt_grid = ParamGridBuilder().addGrid(dt.maxDepth, [15, 20]).addGrid(dt.impurity, ["gini", "entropy"]).build()
start = time.time()
cv_dt = CrossValidator(estimator=pipeline_dt, estimatorParamMaps=dt_grid, evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_dt_model = cv_dt.fit(train_df)
cv_dt_time = time.time() - start

start_pred = time.time()
predictions_dt = cv_dt_model.bestModel.transform(test_df)
dt_pred_time = time.time() - start_pred

metrics_dt = compute_metrics(predictions_dt)
metrics_dt["training_time"] = cv_dt_time
metrics_dt["prediction_time"] = dt_pred_time
metrics_dt["model_size_mb"] = get_model_size(cv_dt_model.bestModel)
cv_results["Decision Tree (Tuned)"] = metrics_dt
report_sections.append({"section_title": "Tuning: Decision Tree", "results": {"Decision Tree (Tuned)": metrics_dt}})

# 2c. GBT
print(f"\n{'━' * 70}\n  2c. Grid Search + CV: GBT\n{'━' * 70}")
gbt = GBTClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
pipeline_gbt = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [gbt])

gbt_grid = ParamGridBuilder().addGrid(gbt.maxIter, [100, 150]).addGrid(gbt.maxDepth, [5, 8]).build()
start = time.time()
cv_gbt = CrossValidator(estimator=pipeline_gbt, estimatorParamMaps=gbt_grid, evaluator=evaluator_cv, numFolds=3, parallelism=2, seed=42)
cv_gbt_model = cv_gbt.fit(train_df)
cv_gbt_time = time.time() - start

start_pred = time.time()
predictions_gbt = cv_gbt_model.bestModel.transform(test_df)
gbt_pred_time = time.time() - start_pred

metrics_gbt = compute_metrics(predictions_gbt)
metrics_gbt["training_time"] = cv_gbt_time
metrics_gbt["prediction_time"] = gbt_pred_time
metrics_gbt["model_size_mb"] = get_model_size(cv_gbt_model.bestModel)
cv_results["GBT (Tuned)"] = metrics_gbt
report_sections.append({"section_title": "Tuning: GBT", "results": {"GBT (Tuned)": metrics_gbt}})

# 2d. Logistic Regression
print(f"\n{'━' * 70}\n  2d. Grid Search + CV: LOGISTIC REGRESSION\n{'━' * 70}")
lr = LogisticRegression(featuresCol=features_col, labelCol="label_binary", family="binomial")
pipeline_lr = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [lr])

lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.elasticNetParam, [0.0, 0.5, 0.8]).build()
start = time.time()
cv_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=lr_grid, evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_lr_model = cv_lr.fit(train_df)
cv_lr_time = time.time() - start

start_pred = time.time()
predictions_lr = cv_lr_model.bestModel.transform(test_df)
lr_pred_time = time.time() - start_pred

metrics_lr = compute_metrics(predictions_lr)
metrics_lr["training_time"] = cv_lr_time
metrics_lr["prediction_time"] = lr_pred_time
metrics_lr["model_size_mb"] = get_model_size(cv_lr_model.bestModel)
cv_results["Logistic Regression (Tuned)"] = metrics_lr
report_sections.append({"section_title": "Tuning: Logistic Regression", "results": {"Logistic Regression (Tuned)": metrics_lr}})


# ==============================================================================
# STEP 3: COMPARE WITH DEFAULT PARAMS
# ==============================================================================
print(f"\n\n{'=' * 70}\n  STEP 3: COMPARE WITH DEFAULT ALGORITHMS\n{'=' * 70}")

default_results, trained_models = run_all_classifiers(
    assembler=assembler_cv, scaler=scaler_cv, train_df=train_df, test_df=test_df,
    features_col=features_col, num_features=num_features, extra_stages=extra_stages,
)

# Merge results
all_results = {}
for name, m in cv_results.items(): all_results[name] = m
for name, m in default_results.items():
    if f"{name} (Tuned)" not in all_results: all_results[f"{name} (Default)"] = m

# Overall Results
print_summary_table(all_results, title=f"GRID SEARCH ON BEST CONFIG: {best_config['method_name']}")

# Plots
plot_comparison(all_results, title="Exp 2: Tuned vs Default", save_path=os.path.join(base_output, "exp2_comparison.png"), show=False)
plot_training_time(all_results, title="Exp 2: Training Time", save_path=os.path.join(base_output, "exp2_train_time.png"), show=False)
plot_roc_curves(trained_models, test_df, title="Exp 2: ROC Curves", save_path=os.path.join(base_output, "exp2_roc_curves.png"), show=False)

report_sections.append({"section_title": "Final Comparison", "results": all_results, "chart_paths": [os.path.join(base_output, "exp2_comparison.png"), os.path.join(base_output, "exp2_roc_curves.png")]})
export_multi_section_report(report_sections, title=f"Exp 2: Grid Search on {best_config['method_name']}", output_path=os.path.join(base_output, "exp2_report.html"))

print("\n[INFO] Experiment 2 completed!")
spark.stop()
