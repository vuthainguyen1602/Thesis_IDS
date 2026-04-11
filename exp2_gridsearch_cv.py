#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from reporting import export_multi_section_report

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
    _get_param,
    _get_best_params,
)

spark = create_spark_session("IDS_Exp2_GridSearch_CV")
df, train_df, test_df, feature_cols = load_and_prepare_data(spark)

# TEMPORARY LIMIT
train_df = train_df.limit(500)
test_df = test_df.limit(500)

print("\n")
print("=" * 70)
print("  EXPERIMENT 2: GRID SEARCH + CROSS-VALIDATION")
print("=" * 70)

print("\n--- Step 1: Loading Best Configuration from Exp 7 ---")

config_path = os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "best_config.json")
best_config = None

if os.path.exists(config_path):
    with open(config_path, "r") as f:
        best_config = json.load(f)
    print(f"[INFO] Loaded best configuration: {best_config['method_name']}")
else:
    print("[WARN] best_config.json not found. Run Exp 7 first.")
    print("Falling back to RF Feature Selection (Top-40) as default.")
    best_config = {
        "method_name": "RF Top-40 (Fallback)",
        "config": {"type": "feature_selection", "csv": os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "feature_importance.csv"), "top_k": 40, "col": "feature"}
    }

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

else:
    print("Using all features")
    assembler_cv = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
    scaler_cv = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)
    features_col = "features_scaled"
    num_features = len(feature_cols)

evaluator_cv = BinaryClassificationEvaluator(
    labelCol="label_binary", rawPredictionCol="rawPrediction", metricName="areaUnderPR",
)


cv_results = {}
cv_models = {}
report_sections = []
base_output = os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "exp2_results")
os.makedirs(base_output, exist_ok=True)

print(f"\n{'━' * 70}\n  2a. Grid Search + CV: RANDOM FOREST\n{'━' * 70}")
rf = RandomForestClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
pipeline_rf = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [rf])

rf_grid = ParamGridBuilder().addGrid(rf.numTrees, [200, 300]).addGrid(rf.maxDepth, [15, 20]).build()
start = time.time()
cv_rf = CrossValidator(estimator=pipeline_rf, estimatorParamMaps=rf_grid, evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_rf_model = cv_rf.fit(train_df)
cv_rf_time = time.time() - start

rf_best_params = _get_best_params(cv_rf_model, rf_grid)
print(f"[BEST] RF: numTrees={rf_best_params.get('numTrees')}, maxDepth={rf_best_params.get('maxDepth')}")

start_pred = time.time()
predictions_rf = cv_rf_model.bestModel.transform(test_df)
rf_pred_time = time.time() - start_pred

metrics_rf = compute_metrics(predictions_rf)
metrics_rf["training_time"] = cv_rf_time
metrics_rf["prediction_time"] = rf_pred_time
metrics_rf["model_size_mb"] = get_model_size(cv_rf_model.bestModel)
cv_results["Random Forest (Tuned)"] = metrics_rf
cv_models["Random Forest (Tuned)"] = cv_rf_model.bestModel
report_sections.append({"section_title": "Tuning: Random Forest", "results": {"Random Forest (Tuned)": metrics_rf}})

print(f"\n{'━' * 70}\n  2b. Grid Search + CV: DECISION TREE\n{'━' * 70}")
dt = DecisionTreeClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
pipeline_dt = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [dt])

dt_grid = ParamGridBuilder().addGrid(dt.maxDepth, [15, 20]).addGrid(dt.impurity, ["gini", "entropy"]).build()
start = time.time()
cv_dt = CrossValidator(estimator=pipeline_dt, estimatorParamMaps=dt_grid, evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_dt_model = cv_dt.fit(train_df)
cv_dt_time = time.time() - start

dt_best = cv_dt_model.bestModel.stages[-1]
print(f"[BEST] DT: maxDepth={_get_param(dt_best, 'maxDepth')}, impurity={_get_param(dt_best, 'impurity')}")

start_pred = time.time()
predictions_dt = cv_dt_model.bestModel.transform(test_df)
dt_pred_time = time.time() - start_pred

metrics_dt = compute_metrics(predictions_dt)
metrics_dt["training_time"] = cv_dt_time
metrics_dt["prediction_time"] = dt_pred_time
metrics_dt["model_size_mb"] = get_model_size(cv_dt_model.bestModel)
cv_results["Decision Tree (Tuned)"] = metrics_dt
cv_models["Decision Tree (Tuned)"] = cv_dt_model.bestModel
report_sections.append({"section_title": "Tuning: Decision Tree", "results": {"Decision Tree (Tuned)": metrics_dt}})

print(f"\n{'━' * 70}\n  2c. Grid Search + CV: GBT\n{'━' * 70}")
gbt = GBTClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
pipeline_gbt = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [gbt])

gbt_grid = ParamGridBuilder().addGrid(gbt.maxIter, [100, 150]).addGrid(gbt.maxDepth, [5, 8]).build()
start = time.time()
cv_gbt = CrossValidator(estimator=pipeline_gbt, estimatorParamMaps=gbt_grid, evaluator=evaluator_cv, numFolds=3, parallelism=2, seed=42)
cv_gbt_model = cv_gbt.fit(train_df)
cv_gbt_time = time.time() - start

gbt_best = cv_gbt_model.bestModel.stages[-1]
print(f"[BEST] GBT: maxIter={_get_param(gbt_best, 'maxIter')}, maxDepth={_get_param(gbt_best, 'maxDepth')}")

start_pred = time.time()
predictions_gbt = cv_gbt_model.bestModel.transform(test_df)
gbt_pred_time = time.time() - start_pred

metrics_gbt = compute_metrics(predictions_gbt)
metrics_gbt["training_time"] = cv_gbt_time
metrics_gbt["prediction_time"] = gbt_pred_time
metrics_gbt["model_size_mb"] = get_model_size(cv_gbt_model.bestModel)
cv_results["GBT (Tuned)"] = metrics_gbt
cv_models["GBT (Tuned)"] = cv_gbt_model.bestModel
report_sections.append({"section_title": "Tuning: GBT", "results": {"GBT (Tuned)": metrics_gbt}})

print(f"\n{'━' * 70}\n  2d. Grid Search + CV: LOGISTIC REGRESSION\n{'━' * 70}")
lr = LogisticRegression(featuresCol=features_col, labelCol="label_binary", family="binomial")
pipeline_lr = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [lr])

lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.elasticNetParam, [0.0, 0.5, 0.8]).build()
start = time.time()
cv_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=lr_grid, evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_lr_model = cv_lr.fit(train_df)
cv_lr_time = time.time() - start

lr_best = cv_lr_model.bestModel.stages[-1]
print(f"[BEST] LR: regParam={_get_param(lr_best, 'regParam')}, elasticNetParam={_get_param(lr_best, 'elasticNetParam')}")

start_pred = time.time()
predictions_lr = cv_lr_model.bestModel.transform(test_df)
lr_pred_time = time.time() - start_pred

metrics_lr = compute_metrics(predictions_lr)
metrics_lr["training_time"] = cv_lr_time
metrics_lr["prediction_time"] = lr_pred_time
metrics_lr["model_size_mb"] = get_model_size(cv_lr_model.bestModel)
cv_results["Logistic Regression (Tuned)"] = metrics_lr
cv_models["Logistic Regression (Tuned)"] = cv_lr_model.bestModel
report_sections.append({"section_title": "Tuning: Logistic Regression", "results": {"Logistic Regression (Tuned)": metrics_lr}})


from shared_utils import HAS_XGBOOST
if HAS_XGBOOST:
    from shared_utils import SparkXGBClassifier
    print(f"\n{'━' * 70}\n  2e. Grid Search + CV: XGBOOST\n{'━' * 70}")
    xgb = SparkXGBClassifier(features_col=features_col, label_col="label_binary", num_workers=4)
    pipeline_xgb = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [xgb])
    xgb_grid = ParamGridBuilder().addGrid(xgb.max_depth, [5, 8]).addGrid(xgb.learning_rate, [0.05, 0.1]).build()
    start = time.time()
    cv_xgb = CrossValidator(estimator=pipeline_xgb, estimatorParamMaps=xgb_grid, evaluator=evaluator_cv, numFolds=3, parallelism=2, seed=42)
    cv_xgb_model = cv_xgb.fit(train_df)
    cv_xgb_time = time.time() - start

    xgb_best_params = _get_best_params(cv_xgb_model, xgb_grid)
    print(f"[BEST] XGBoost: max_depth={xgb_best_params.get('max_depth')}, learning_rate={xgb_best_params.get('learning_rate')}")
    
    start_pred = time.time()
    predictions_xgb = cv_xgb_model.bestModel.transform(test_df)
    xgb_pred_time = time.time() - start_pred
    
    metrics_xgb = compute_metrics(predictions_xgb)
    metrics_xgb["training_time"] = cv_xgb_time
    metrics_xgb["prediction_time"] = xgb_pred_time
    metrics_xgb["model_size_mb"] = get_model_size(cv_xgb_model.bestModel)
    cv_results["XGBoost (Tuned)"] = metrics_xgb
    cv_models["XGBoost (Tuned)"] = cv_xgb_model.bestModel
    report_sections.append({"section_title": "Tuning: XGBoost", "results": {"XGBoost (Tuned)": metrics_xgb}})

from shared_utils import HAS_LIGHTGBM
if HAS_LIGHTGBM:
    from shared_utils import LightGBMClassifier
    print(f"\n{'━' * 70}\n  2f. Grid Search + CV: LIGHTGBM\n{'━' * 70}")
    lgbm = LightGBMClassifier(featuresCol=features_col, labelCol="label_binary", objective="binary")
    pipeline_lgbm = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [lgbm])
    lgbm_grid = ParamGridBuilder().addGrid(lgbm.numLeaves, [31, 63]).addGrid(lgbm.learningRate, [0.05, 0.1]).build()
    start = time.time()
    cv_lgbm = CrossValidator(estimator=pipeline_lgbm, estimatorParamMaps=lgbm_grid, evaluator=evaluator_cv, numFolds=3, parallelism=2, seed=42)
    cv_lgbm_model = cv_lgbm.fit(train_df)
    cv_lgbm_time = time.time() - start

    lgbm_best_params = _get_best_params(cv_lgbm_model, lgbm_grid)
    print(f"[BEST] LightGBM: numLeaves={lgbm_best_params.get('numLeaves')}, learningRate={lgbm_best_params.get('learningRate')}")
    
    start_pred = time.time()
    predictions_lgbm = cv_lgbm_model.bestModel.transform(test_df)
    lgbm_pred_time = time.time() - start_pred
    
    metrics_lgbm = compute_metrics(predictions_lgbm)
    metrics_lgbm["training_time"] = cv_lgbm_time
    metrics_lgbm["prediction_time"] = lgbm_pred_time
    metrics_lgbm["model_size_mb"] = get_model_size(cv_lgbm_model.bestModel)
    cv_results["LightGBM (Tuned)"] = metrics_lgbm
    cv_models["LightGBM (Tuned)"] = cv_lgbm_model.bestModel
    report_sections.append({"section_title": "Tuning: LightGBM", "results": {"LightGBM (Tuned)": metrics_lgbm}})

from shared_utils import MultilayerPerceptronClassifier
print(f"\n{'━' * 70}\n  2g. Grid Search + CV: MLP\n{'━' * 70}")
layers_opts = [[num_features, 64, 32, 2], [num_features, 128, 64, 32, 2]]
mlp = MultilayerPerceptronClassifier(featuresCol=features_col, labelCol="label_binary", seed=42)
pipeline_mlp = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [mlp])
mlp_grid = ParamGridBuilder().addGrid(mlp.layers, layers_opts).addGrid(mlp.maxIter, [100, 200]).build()
start = time.time()
cv_mlp = CrossValidator(estimator=pipeline_mlp, estimatorParamMaps=mlp_grid, evaluator=evaluator_cv, numFolds=3, parallelism=4, seed=42)
cv_mlp_model = cv_mlp.fit(train_df)
cv_mlp_time = time.time() - start

mlp_best = cv_mlp_model.bestModel.stages[-1]
print(f"[BEST] MLP: layers={_get_param(mlp_best, 'layers')}, maxIter={_get_param(mlp_best, 'maxIter')}")

start_pred = time.time()
predictions_mlp = cv_mlp_model.bestModel.transform(test_df)
mlp_pred_time = time.time() - start_pred

metrics_mlp = compute_metrics(predictions_mlp)
metrics_mlp["training_time"] = cv_mlp_time
metrics_mlp["prediction_time"] = mlp_pred_time
metrics_mlp["model_size_mb"] = get_model_size(cv_mlp_model.bestModel)
cv_results["MLP (Tuned)"] = metrics_mlp
cv_models["MLP (Tuned)"] = cv_mlp_model.bestModel
report_sections.append({"section_title": "Tuning: MLP", "results": {"MLP (Tuned)": metrics_mlp}})

print(f"\n{'━' * 70}\n  2h. Ensemble Voting (Tuned Models)\n{'━' * 70}")
from shared_utils import ensemble_voting
ens_metrics = ensemble_voting(cv_models, test_df, results=cv_results)
if ens_metrics:
    cv_results["Ensemble Voting (Tuned)"] = ens_metrics
    report_sections.append({"section_title": "Tuning: Ensemble", "results": {"Ensemble Voting (Tuned)": ens_metrics}})

print(f"\n{'━' * 70}\n  2i. Hybrid Bagging Ensemble (Tuned Models)\n{'━' * 70}")
from shared_utils import train_hybrid_bagging

# Lấy best params trực tiếp từ avgMetrics index - đáng tin cậy 100%
rf_tuned = RandomForestClassifier(
    featuresCol=features_col, labelCol="label_binary",
    numTrees=int(rf_best_params.get("numTrees", 200)),
    maxDepth=int(rf_best_params.get("maxDepth", 15)),
    seed=42,
)
pipeline_rf_t = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [rf_tuned])

pipeline_dist_tuned = [(pipeline_rf_t, 3)]

if HAS_XGBOOST:
    from shared_utils import SparkXGBClassifier
    xgb_tuned = SparkXGBClassifier(
        features_col=features_col, label_col="label_binary",
        max_depth=int(xgb_best_params.get("max_depth", 8)),
        learning_rate=float(xgb_best_params.get("learning_rate", 0.05)),
        num_workers=4,
    )
    pipeline_xgb_t = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [xgb_tuned])
    pipeline_dist_tuned.append((pipeline_xgb_t, 2))

if HAS_LIGHTGBM:
    from shared_utils import LightGBMClassifier
    lgbm_tuned = LightGBMClassifier(
        featuresCol=features_col, labelCol="label_binary",
        numLeaves=int(lgbm_best_params.get("numLeaves", 63)),
        learningRate=float(lgbm_best_params.get("learningRate", 0.05)),
        objective="binary",
    )
    pipeline_lgbm_t = Pipeline(stages=[assembler_cv, scaler_cv] + extra_stages + [lgbm_tuned])
    pipeline_dist_tuned.append((pipeline_lgbm_t, 2))

bag_model_tuned = train_hybrid_bagging(pipeline_dist_tuned, train_df)
if bag_model_tuned:
    start_pred = time.time()
    bag_preds_tuned = bag_model_tuned.transform(test_df)
    bag_preds_tuned.cache().count()
    bag_pred_time_tuned = time.time() - start_pred
    
    metrics_bag_tuned = compute_metrics(bag_preds_tuned)
    metrics_bag_tuned["training_time"] = 0.0
    metrics_bag_tuned["prediction_time"] = bag_pred_time_tuned
    
    total_size_bag = 0.0
    for m in bag_model_tuned.models:
        total_size_bag += get_model_size(m)
    metrics_bag_tuned["model_size_mb"] = total_size_bag
    
    cv_results["Hybrid Bagging Ensemble (Tuned)"] = metrics_bag_tuned
    report_sections.append({"section_title": "Tuning: Hybrid Bagging", "results": {"Hybrid Bagging Ensemble (Tuned)": metrics_bag_tuned}})


print(f"\n\n{'=' * 70}\n  STEP 3: CONSOLIDATED TUNED RESULTS\n{'=' * 70}")

all_results = cv_results

print_summary_table(all_results, title=f"GRID SEARCH ON BEST CONFIG: {best_config['method_name']}")

plot_comparison(all_results, title="Exp 2: Tuned Models Comparison", save_path=os.path.join(base_output, "exp2_comparison.png"), show=False)
plot_training_time(all_results, title="Exp 2: Tuning Time", save_path=os.path.join(base_output, "exp2_train_time.png"), show=False)

plot_roc_curves(cv_models, test_df, title="Exp 2: Tuned Models ROC Curves", save_path=os.path.join(base_output, "exp2_roc_curves.png"), show=False)

report_sections.append({
    "section_title": "Final Comparison (Tuned Models)",
    "results": all_results,
    "chart_paths": [
        os.path.join(base_output, "exp2_comparison.png"),
        os.path.join(base_output, "exp2_roc_curves.png")
    ]
})

export_multi_section_report(report_sections, title=f"Exp 2: Grid Search on {best_config['method_name']}", output_path=os.path.join(base_output, "exp2_report.html"))

print("\n[INFO] Experiment 2 completed!")
spark.stop()
