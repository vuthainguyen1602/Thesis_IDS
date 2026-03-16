#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
EXPERIMENT 2: HYPERPARAMETER OPTIMIZATION - GRID SEARCH + CV (RoEduNet)
================================================================================
Hyperparameter optimization using Grid Search + Cross-Validation.
Grid Search + CV applied to: Random Forest, GBT, Decision Tree, Logistic Regression.
Results compared with all algorithms including XGBoost, LightGBM, MLP.
================================================================================
"""

import os
import time
import pandas as pd


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
)

# ==============================================================================
# INITIALIZATION
# ==============================================================================
spark = create_spark_session("RoEduNet_Exp2_GridSearch_CV")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

print("\n")
print("█" * 70)
print("  EXPERIMENT 2: GRID SEARCH + CV (RoEduNet-SIMARGL2021)")
print("█" * 70)

# ==============================================================================
# STEP 1: PREPARE FEATURES (TOP-15 FROM RF IMPORTANCE)
# ==============================================================================
print("\n--- Step 1: Prepare features ---")

feature_importance_path = "/Users/thainguyenvu/Desktop/Thesis_IDS/Thesis_IDS_RoEduNet/feature_importance.csv"

try:
    feature_importance_df = pd.read_csv(feature_importance_path)
    print("✓ Read feature_importance.csv successfully")
except FileNotFoundError:
    print("Running RF Baseline to generate feature_importance.csv...")
    assembler_tmp = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
    scaler_tmp = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
    rf_tmp = RandomForestClassifier(featuresCol="features_scaled", labelCol="label_binary", numTrees=100, maxDepth=10, seed=42)
    pipeline_tmp = Pipeline(stages=[assembler_tmp, scaler_tmp, rf_tmp])
    model_tmp = pipeline_tmp.fit(train_df)
    importances = model_tmp.stages[-1].featureImportances.toArray()
    feature_importance_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feature_importance_df = feature_importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    feature_importance_df.to_csv(feature_importance_path, index=False)

best_top_k = 15
selected_features = feature_importance_df.head(best_top_k)["feature"].tolist()
print(f"Using Top-{best_top_k} features")

assembler_cv = VectorAssembler(inputCols=selected_features, outputCol="features_raw", handleInvalid="keep")
scaler_cv = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)

# ★ Use PR-AUC as CV evaluation metric (correlates with F1 for binary classification)
evaluator_cv = BinaryClassificationEvaluator(
    labelCol="label_binary", rawPredictionCol="rawPrediction", metricName="areaUnderPR",
)
print("★ CV Evaluation Metric: PR-AUC (Precision-Recall AUC → optimizes F1)")


# ==============================================================================
# STEP 2: HYPERPARAMETER OPTIMIZATION (GRID SEARCH + CV)
# ==============================================================================

cv_results = {}
report_sections = []

base_output = "/Users/thainguyenvu/Desktop/Thesis_IDS/Thesis_IDS_RoEduNet/exp2_results"
os.makedirs(base_output, exist_ok=True)

# ─────────────────────────────────────────────
# 2a. Random Forest
# ─────────────────────────────────────────────
print(f"\n{'━' * 70}")
print("  2a. Grid Search + CV: RANDOM FOREST")
print(f"{'━' * 70}")

rf = RandomForestClassifier(featuresCol="features_scaled", labelCol="label_binary", seed=42)
pipeline_rf = Pipeline(stages=[assembler_cv, scaler_cv, rf])

rf_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [200, 300])
    .addGrid(rf.maxDepth, [15, 20])
    .build()
)
print(f"Total combinations: {len(rf_grid)}")

start = time.time()
cv_rf = CrossValidator(estimator=pipeline_rf, estimatorParamMaps=rf_grid,
                        evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_rf_model = cv_rf.fit(train_df)
cv_rf_time = time.time() - start

best_rf = cv_rf_model.bestModel.stages[-1]
print(f"\nBest hyperparameters:")
print(f"  numTrees: {best_rf.getNumTrees}, maxDepth: {best_rf.getOrDefault('maxDepth')}")
print(f"  Best CV PR-AUC: {max(cv_rf_model.avgMetrics):.6f}")

preds_rf = cv_rf_model.bestModel.transform(test_df)
metrics_rf = compute_metrics(preds_rf)
metrics_rf["training_time"] = cv_rf_time
print_metrics(metrics_rf, "RF after Grid Search + CV")
cv_results["Random Forest (Tuned)"] = metrics_rf

report_sections.append({
    "section_title": "Hyperparameter Tuning: Random Forest",
    "results": {"Random Forest (Tuned)": metrics_rf},
})

# ─────────────────────────────────────────────
# 2b. Decision Tree
# ─────────────────────────────────────────────
print(f"\n{'━' * 70}")
print("  2b. Grid Search + CV: DECISION TREE")
print(f"{'━' * 70}")

dt = DecisionTreeClassifier(featuresCol="features_scaled", labelCol="label_binary", seed=42)
pipeline_dt = Pipeline(stages=[assembler_cv, scaler_cv, dt])

dt_grid = (
    ParamGridBuilder()
    .addGrid(dt.maxDepth, [15, 20])
    .addGrid(dt.impurity, ["gini", "entropy"])
    .build()
)
print(f"Total combinations: {len(dt_grid)}")

start = time.time()
cv_dt = CrossValidator(estimator=pipeline_dt, estimatorParamMaps=dt_grid,
                        evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_dt_model = cv_dt.fit(train_df)
cv_dt_time = time.time() - start

best_dt = cv_dt_model.bestModel.stages[-1]
print(f"\nBest hyperparameters:")
print(f"  maxDepth: {best_dt.getOrDefault('maxDepth')}, impurity: {best_dt.getOrDefault('impurity')}")

preds_dt = cv_dt_model.bestModel.transform(test_df)
metrics_dt = compute_metrics(preds_dt)
metrics_dt["training_time"] = cv_dt_time
print_metrics(metrics_dt, "DT after Grid Search + CV")
cv_results["Decision Tree (Tuned)"] = metrics_dt

report_sections.append({
    "section_title": "Hyperparameter Tuning: Decision Tree",
    "results": {"Decision Tree (Tuned)": metrics_dt},
})

# ─────────────────────────────────────────────
# 2c. Gradient Boosted Trees
# ─────────────────────────────────────────────
print(f"\n{'━' * 70}")
print("  2c. Grid Search + CV: GBT")
print(f"{'━' * 70}")

gbt = GBTClassifier(featuresCol="features_scaled", labelCol="label_binary", seed=42)
pipeline_gbt = Pipeline(stages=[assembler_cv, scaler_cv, gbt])

gbt_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxIter, [150, 200])
    .addGrid(gbt.maxDepth, [6, 8])
    .build()
)
print(f"Total combinations: {len(gbt_grid)}")

start = time.time()
cv_gbt = CrossValidator(estimator=pipeline_gbt, estimatorParamMaps=gbt_grid,
                         evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_gbt_model = cv_gbt.fit(train_df)
cv_gbt_time = time.time() - start

best_gbt = cv_gbt_model.bestModel.stages[-1]
print(f"\nBest hyperparameters:")
print(f"  maxIter: {best_gbt.getOrDefault('maxIter')}, maxDepth: {best_gbt.getOrDefault('maxDepth')}, stepSize: {best_gbt.getOrDefault('stepSize')}")

preds_gbt = cv_gbt_model.bestModel.transform(test_df)
metrics_gbt = compute_metrics(preds_gbt)
metrics_gbt["training_time"] = cv_gbt_time
print_metrics(metrics_gbt, "GBT after Grid Search + CV")
cv_results["GBT (Tuned)"] = metrics_gbt

report_sections.append({
    "section_title": "Hyperparameter Tuning: GBT",
    "results": {"GBT (Tuned)": metrics_gbt},
})

# ─────────────────────────────────────────────
# 2d. Logistic Regression
# ─────────────────────────────────────────────
print(f"\n{'━' * 70}")
print("  2d. Grid Search + CV: LOGISTIC REGRESSION")
print(f"{'━' * 70}")

lr = LogisticRegression(featuresCol="features_scaled", labelCol="label_binary", family="binomial")
pipeline_lr = Pipeline(stages=[assembler_cv, scaler_cv, lr])

lr_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.001, 0.01])
    .addGrid(lr.elasticNetParam, [0.3, 0.5])
    .build()
)
print(f"Total combinations: {len(lr_grid)}")

start = time.time()
cv_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=lr_grid,
                        evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_lr_model = cv_lr.fit(train_df)
cv_lr_time = time.time() - start

best_lr = cv_lr_model.bestModel.stages[-1]
print(f"\nBest hyperparameters:")
print(f"  regParam: {best_lr.getOrDefault('regParam')}, elasticNetParam: {best_lr.getOrDefault('elasticNetParam')}")

preds_lr = cv_lr_model.bestModel.transform(test_df)
metrics_lr = compute_metrics(preds_lr)
metrics_lr["training_time"] = cv_lr_time
print_metrics(metrics_lr, "LR after Grid Search + CV")
cv_results["Logistic Regression (Tuned)"] = metrics_lr

report_sections.append({
    "section_title": "Hyperparameter Tuning: Logistic Regression",
    "results": {"Logistic Regression (Tuned)": metrics_lr},
})


# ==============================================================================
# STEP 3: COMPARE TUNED MODELS WITH OTHER ALGORITHMS (DEFAULT PARAMS)
# ==============================================================================
print(f"\n\n{'█' * 70}")
print("  STEP 3: COMPARE WITH OTHER ALGORITHMS (DEFAULT PARAMS)")
print(f"{'█' * 70}")

default_results, trained_models = run_all_classifiers(
    assembler=assembler_cv,
    scaler=scaler_cv,
    train_df=train_df,
    test_df=test_df,
    features_col="features_scaled",
    num_features=best_top_k,
)

# Ensemble Voting
ens_metrics = ensemble_voting(trained_models, test_df, results=results)
if ens_metrics:
    default_results["Ensemble Voting"] = ens_metrics

# Merge results: Tuned models + Default models
all_results = {}
for name, m in cv_results.items():
    all_results[name] = m
for name, m in default_results.items():
    base_name = name.replace(" (Tuned)", "")
    if f"{base_name} (Tuned)" not in all_results:
        all_results[f"{name} (Default)"] = m


# ==============================================================================
# OVERALL RESULTS
# ==============================================================================
print_summary_table(all_results, title="EXPERIMENT 2 SUMMARY: GRID SEARCH + CV (RoEduNet)")

plot_comparison(
    all_results,
    title="RoEduNet Exp 2: Tuned vs Default Comparison",
    save_path=os.path.join(base_output, "exp2_comparison.png"),
    show=False,
)
plot_training_time(
    all_results,
    title="RoEduNet Exp 2: Training Time Comparison",
    save_path=os.path.join(base_output, "exp2_train_time.png"),
    show=False,
)
plot_prediction_time(
    all_results,
    title="RoEduNet Exp 2: Prediction Time Comparison",
    save_path=os.path.join(base_output, "exp2_pred_time.png"),
    show=False,
)
plot_model_size(
    all_results,
    title="RoEduNet Exp 2: Model Size Comparison",
    save_path=os.path.join(base_output, "exp2_model_size.png"),
    show=False,
)
plot_confusion_matrices(
    all_results,
    title="RoEduNet Exp 2: Confusion Matrices",
    save_path=os.path.join(base_output, "exp2_confusion_matrices.png"),
    show=False,
)
plot_roc_curves(
    trained_models, test_df,
    title="RoEduNet Exp 2: ROC Curves",
    save_path=os.path.join(base_output, "exp2_roc_curves.png"),
    show=False,
)

report_sections.append({
    "section_title": "Final Comparison: Tuned vs Default Models",
    "results": all_results,
    "chart_paths": [
        os.path.join(base_output, "exp2_comparison.png"),
        os.path.join(base_output, "exp2_train_time.png"),
        os.path.join(base_output, "exp2_pred_time.png"),
        os.path.join(base_output, "exp2_model_size.png"),
        os.path.join(base_output, "exp2_confusion_matrices.png"),
        os.path.join(base_output, "exp2_roc_curves.png"),
    ],
})

export_multi_section_report(
    report_sections,
    title="RoEduNet - Experiment 2: Hyperparameter Optimization",
    output_path=os.path.join(base_output, "exp2_report.html"),
)

print("\n✓ Experiment 2 (RoEduNet Grid Search + CV) completed!")
spark.stop()
print("✓ Spark Session closed.")
