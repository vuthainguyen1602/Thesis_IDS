#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

logger = logging.getLogger(__name__)

import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[4] pyspark-shell'

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, monotonically_increasing_id, udf, array
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import (
    RandomForestClassifier,
    DecisionTreeClassifier,
    LogisticRegression,
    GBTClassifier,
    LinearSVC,
    NaiveBayes,
    MultilayerPerceptronClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

try:
    from xgboost.spark import SparkXGBClassifier
    HAS_XGBOOST = True
    print("[INFO] XGBoost backend available")
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not available (pip install xgboost)")

try:
    from synapse.ml.lightgbm import LightGBMClassifier
    _test = LightGBMClassifier()
    del _test
    HAS_LIGHTGBM = True
    print("[INFO] LightGBM backend available")
except (ImportError, TypeError, Exception):
    HAS_LIGHTGBM = False
    print("[WARN] LightGBM not available (missing Java backend or pip install synapseml)")


_DEFAULT_DATA_DIR: str = os.environ.get(
    "IDS_DATA_DIR",
    os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "data")
)


def create_spark_session(app_name: str = "IDS_Binary_Prediction") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "8g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "100s")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"[INFO] Spark {spark.version} | UI: {spark.sparkContext.uiWebUrl}")
    return spark


def clean_column_names(df):
    for col_name in df.columns:
        new_name = col_name.strip().lower()
        new_name = re.sub(r"[\s.\-/]+", "_", new_name)
        new_name = re.sub(r"[()]", "", new_name)
        new_name = re.sub(r"_+", "_", new_name)
        new_name = new_name.strip("_")
        df = df.withColumnRenamed(col_name, new_name)
    return df


def handle_infinity_values(df):
    for col_name in df.columns:
        if dict(df.dtypes)[col_name] in ["double", "float"]:
            df = df.withColumn(
                col_name,
                F.when(
                    (F.col(col_name).isNull())
                    | (F.isnan(F.col(col_name)))
                    | (F.col(col_name) == float("inf"))
                    | (F.col(col_name) == float("-inf")),
                    None,
                ).otherwise(F.col(col_name)),
            )
    return df


def align_schema(df, ref_columns: list):
    for c in ref_columns:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(StringType()))
    return df.select(ref_columns)


def load_and_prepare_data(
    spark, data_dir: str = _DEFAULT_DATA_DIR
) -> tuple:
    output_dir = data_dir
    train_path = os.path.join(output_dir, "train_data.parquet")
    test_path = os.path.join(output_dir, "test_data.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Parquet data not found. Run data_preparation.py first!\n"
            f"  Expected: {train_path}\n"
            f"  Expected: {test_path}"
        )

    print("=" * 60)
    print("  LOADING DATA FROM PARQUET")
    print("=" * 60)

    train_df = spark.read.parquet(train_path)
    test_df = spark.read.parquet(test_path)

    train_df = train_df.cache()
    test_df = test_df.cache()

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"  Training set: {train_count:,} samples")
    print(f"  Test set:     {test_count:,} samples")

    df = train_df.unionByName(test_df).cache()

    exclude_cols = ["label", "label_binary", "source_ip", "destination_ip",
                    "flow_id", "timestamp", "protocol"]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and dict(df.dtypes)[c] in ["double", "float", "int", "bigint"]
    ]
    print(f"  Numeric features: {len(feature_cols)}")
    print("=" * 60)

    return df, train_df, test_df, feature_cols


def get_classifiers(
    features_col: str,
    label_col: str = "label_binary",
    num_features: int = 50,
    scale_pos_weight: float = None,
) -> OrderedDict:
    classifiers = OrderedDict()

    classifiers["Decision Tree"] = DecisionTreeClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        maxDepth=15,
        minInstancesPerNode=5,
        impurity="entropy",
        seed=42,
    )

    classifiers["Logistic Regression"] = LogisticRegression(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=200,
        regParam=0.001,
        elasticNetParam=0.0,
        family="binomial",
        threshold=0.4,
    )

    classifiers["SVM (LinearSVC)"] = LinearSVC(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=100,
        regParam=0.001,
        threshold=0.0,
    )

    classifiers["Naive Bayes"] = NaiveBayes(
        featuresCol=features_col,
        labelCol=label_col,
        modelType="gaussian",
        smoothing=1.0,
    )

    classifiers["Random Forest"] = RandomForestClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        numTrees=200,
        maxDepth=15,
        minInstancesPerNode=5,
        featureSubsetStrategy="sqrt",
        subsamplingRate=1.0,
        seed=42,
    )

    classifiers["GBT"] = GBTClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=150,
        maxDepth=6,
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=42,
    )

    if HAS_XGBOOST:
        xgb_params = dict(
            features_col=features_col,
            label_col=label_col,
            num_workers=4,
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.05,
            reg_alpha=0.5,
            reg_lambda=1.0,
            use_gpu=False,
        )
        if scale_pos_weight is not None:
            xgb_params["scale_pos_weight"] = scale_pos_weight
        classifiers["XGBoost"] = SparkXGBClassifier(**xgb_params)

    if HAS_LIGHTGBM:
        classifiers["LightGBM"] = LightGBMClassifier(
            featuresCol=features_col,
            labelCol=label_col,
            numIterations=300,
            numLeaves=63,
            maxDepth=8,
            learningRate=0.05,
            minDataInLeaf=20,
            featureFraction=0.8,
            baggingFraction=0.8,
            baggingFreq=5,
            lambdaL1=0.5,
            lambdaL2=1.0,
            objective="binary",
        )

    layers = [num_features, 128, 64, 32, 2]
    classifiers["MLP"] = MultilayerPerceptronClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        layers=layers,
        maxIter=200,
        blockSize=128,
        stepSize=0.01,
        seed=42,
    )

    return classifiers


def compute_metrics(
    predictions,
    label_col: str = "label_binary",
) -> dict:
    counts = predictions.groupBy(label_col, "prediction").count().collect()
    TP = TN = FP = FN = 0
    for row in counts:
        l = row[label_col]
        p = row["prediction"]
        c = row["count"]
        if l == 1 and p == 1: TP = c
        elif l == 0 and p == 0: TN = c
        elif l == 0 and p == 1: FP = c
        elif l == 1 and p == 0: FN = c

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    auc_roc = None
    auc_pr = None
    
    score_col = None
    if "rawPrediction" in predictions.columns:
        score_col = "rawPrediction"
    elif "avg_probability" in predictions.columns:
        score_col = "avg_probability"
    elif "probability" in predictions.columns:
        score_col = "probability"

    if score_col:
        try:
            evaluator_roc = BinaryClassificationEvaluator(
                labelCol=label_col, rawPredictionCol=score_col,
                metricName="areaUnderROC"
            )
            auc_roc = evaluator_roc.evaluate(predictions)

            evaluator_pr = BinaryClassificationEvaluator(
                labelCol=label_col, rawPredictionCol=score_col,
                metricName="areaUnderPR"
            )
            auc_pr = evaluator_pr.evaluate(predictions)
        except Exception:
            pass

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }


def print_metrics(metrics: dict, title: str = "") -> None:
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {title}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {metrics['accuracy']:.6f}")
    print(f"  Precision: {metrics['precision']:.6f}")
    print(f"  Recall:    {metrics['recall']:.6f}")
    print(f"  F1-Score:  {metrics['f1']:.6f}")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.6f}")
    if metrics.get("auc_pr") is not None:
        print(f"  AUC-PR:    {metrics['auc_pr']:.6f}")
    print(f"  TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    if metrics.get("model_size_mb") is not None:
        print(f"  Model Size: {metrics['model_size_mb']:.3f} MB")
    if metrics.get("training_time") is not None:
        print(f"  Training time:    {metrics['training_time']:.3f}s")
    if metrics.get("prediction_time") is not None:
        print(f"  Prediction time:  {metrics['prediction_time']:.3f}s")
    print(f"{'=' * 60}")


def get_model_size(model) -> float:
    import shutil
    temp_path = "/tmp/spark_model_size_temp"
    try:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        model.save(temp_path)

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(temp_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

        return total_size / (1024 * 1024)
    except Exception:
        return 0.0
    finally:
        if os.path.exists(temp_path):
            try:
                shutil.rmtree(temp_path)
            except Exception:
                pass


def _get_param(model, param_name: str, default=None):
    """
    Robust parameter extraction from PySpark, XGBoost, or SynapseML models.
    Tries multiple naming conventions and fallback mechanisms.
    """
    # 1. Try standard getter (e.g., getNumTrees)
    getter = "get" + param_name[0].upper() + param_name[1:]
    if hasattr(model, getter):
        try:
            val = getattr(model, getter)()
            if val is not None: return val
        except Exception:
            pass

    # 2. Try variations of the name (camelCase vs snake_case)
    variations = [param_name]
    if "_" in param_name:
        parts = param_name.split("_")
        variations.append(parts[0] + "".join(p.capitalize() for p in parts[1:]))
    else:
        import re
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", param_name)
        variations.append(re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower())

    # Remove duplicates
    variations = list(dict.fromkeys(variations))

    # 3. Try extractParamMap
    try:
        pmap = {p.name: v for p, v in model.extractParamMap().items()}
        for var in variations:
            if var in pmap:
                return pmap[var]
    except Exception:
        pass

    # 4. Try getOrDefault with variations
    for var in variations:
        try:
            val = model.getOrDefault(var)
            if val is not None: return val
        except Exception:
            pass

    # 5. Try direct attribute access
    for var in variations:
        if hasattr(model, var):
            val = getattr(model, var)
            if "pyspark.ml.param.Param" not in str(type(val)):
                return val

    return default if default is not None else "N/A"


def _get_best_params(cv_model, param_grid):
    best_idx = int(np.argmax(cv_model.avgMetrics))
    best_param_map = param_grid[best_idx]
    return {p.name: v for p, v in best_param_map.items()}


def train_and_evaluate(pipeline, train_df, test_df, title: str = "") -> tuple:
    start = time.time()
    model = pipeline.fit(train_df)
    training_time = time.time() - start

    start_pred = time.time()
    predictions = model.transform(test_df)
    prediction_time = time.time() - start_pred

    metrics = compute_metrics(predictions)
    metrics["training_time"] = training_time
    metrics["prediction_time"] = prediction_time
    metrics["model_size_mb"] = get_model_size(model)

    print_metrics(metrics, title)

    return model, predictions, metrics


class BaggingModel:
    def __init__(self, models, names=None, weights=None):
        self.models = models
        self.names = names or [f"Model_{i}" for i in range(len(models))]
        if weights is not None:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

    def transform(self, df):
        from pyspark.sql.functions import col, when, monotonically_increasing_id, udf
        from pyspark.sql.types import DoubleType

        df_with_id = df.withColumn("_row_id", monotonically_increasing_id()).cache()
        df_with_id.count()
        
        extract_prob_udf = udf(lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0, DoubleType())
        vector_prob_udf = udf(lambda p: Vectors.dense([1.0 - p, p]), VectorUDT())
        
        combined_result = df_with_id.select("_row_id", "label_binary")
        
        for i, model in enumerate(self.models):
            preds = model.transform(df_with_id)
            preds = preds.withColumn(f"prob_{i}", extract_prob_udf(col("probability")))
            combined_result = combined_result.join(preds.select("_row_id", f"prob_{i}"), on="_row_id")
            
            if (i + 1) % 5 == 0:
                combined_result = combined_result.localCheckpoint()

        weighted_probs = []
        for i in range(len(self.models)):
            weighted_probs.append(col(f"prob_{i}") * self.weights[i])
        
        avg_prob_expr = sum(weighted_probs)
        
        final_res = combined_result.withColumn("avg_probability", avg_prob_expr)
        final_res = final_res.withColumn("rawPrediction", col("avg_probability"))
        final_res = final_res.withColumn("probability", vector_prob_udf(col("avg_probability")))
        
        final_res = final_res.withColumn(
            "prediction", 
            when(col("avg_probability") >= 0.5, 1.0).otherwise(0.0)
        )
        
        return final_res

    def save(self, path: str) -> None:
        raise NotImplementedError(
            "BaggingModel.save() is not yet implemented. "
            "Serialise individual base models via model.save(path) instead."
        )

def train_manual_bagging(
    base_pipeline, train_df, num_models: int = 5,
    label_col: str = "label_binary", benign_ratio: float = 0.7,
) -> BaggingModel:
    models = []
    print(f"  Training Balanced Bagging Ensemble ({num_models} models, {benign_ratio*100:.0f}/{ (1-benign_ratio)*100:.0f} ratio)...")
    
    attack_df = train_df.filter(col(label_col) == 1).cache()
    benign_df = train_df.filter(col(label_col) == 0).cache()
    
    attack_count = attack_df.count()
    benign_count = benign_df.count()
    
    if attack_count == 0 or benign_count == 0:
        return None

    target_benign_count = int((benign_ratio * attack_count) / (1 - benign_ratio))
    benign_fraction = float(target_benign_count) / benign_count
    
    for i in range(num_models):
        print(f"   - Building bootstrap model {i+1}/{num_models}...")
        
        attack_sample = attack_df.sample(withReplacement=True, fraction=1.0, seed=42 + i)
        
        benign_sample = benign_df.sample(withReplacement=True, fraction=min(benign_fraction, 10.0), seed=100 + i)
        
        balanced_bag = attack_sample.unionAll(benign_sample)
        
        model = base_pipeline.fit(balanced_bag)
        models.append(model)
        
    attack_df.unpersist()
    benign_df.unpersist()
    
    return BaggingModel(models)

def train_hybrid_bagging(
    pipeline_distribution, train_df,
    label_col: str = "label_binary",
    benign_ratio: float = 0.7, balanced: bool = True,
    feature_list: list = None, feature_subset_rate: float = 1.0,
) -> BaggingModel:
    total_models = sum([count for _, count in pipeline_distribution])
    models = []
    
    if balanced:
        attack_df = train_df.filter(col(label_col) == 1.0).cache()
        benign_df = train_df.filter(col(label_col) == 0.0).cache()
        
        attack_count = attack_df.count()
        benign_count = benign_df.count()
        
        target_benign_count = int((benign_ratio * attack_count) / (1 - benign_ratio))
        benign_fraction = float(target_benign_count) / benign_count
    else:
        full_train_df = train_df.cache()

    current_idx = 0
    for base_pipeline, num_replicas in pipeline_distribution:
        for i in range(num_replicas):
            current_idx += 1
            print(f"   - Building hybrid model {current_idx}/{total_models} (Balanced={balanced})...")
            
            if balanced:
                attack_sample = attack_df.sample(withReplacement=True, fraction=1.0, seed=42 + current_idx)
                benign_sample = benign_df.sample(withReplacement=True, fraction=min(benign_fraction, 10.0), seed=100 + current_idx)
                bag_df = attack_sample.unionAll(benign_sample)
            else:
                bag_df = full_train_df.sample(withReplacement=True, fraction=1.0, seed=42 + current_idx)
            
            bag_pipeline = base_pipeline.copy()
            
            if feature_list and feature_subset_rate < 1.0:
                k = max(1, int(len(feature_list) * feature_subset_rate))
                random.seed(42 + current_idx)
                selected_features = random.sample(feature_list, k)
                print(f"     * Feature Bagging: {k}/{len(feature_list)} random features selected")
                
                stages = bag_pipeline.getStages()
                for stage in stages:
                    if isinstance(stage, VectorAssembler):
                        stage.setInputCols(selected_features)
                        break
                bag_pipeline.setStages(stages)
            
            model = bag_pipeline.fit(bag_df)
            models.append(model)
            
    if balanced:
        attack_df.unpersist()
        benign_df.unpersist()
    else:
        full_train_df.unpersist()
    
    return BaggingModel(models)


def run_all_classifiers(
    assembler,
    scaler,
    train_df,
    test_df,
    features_col: str,
    num_features: int,
    label_col: str = "label_binary",
    extra_stages=None,
) -> tuple:
    class_counts = (
        train_df.groupBy(label_col).count().collect()
    )
    count_map = {row[label_col]: row["count"] for row in class_counts}
    benign_count = count_map.get(0, 0)
    attack_count = count_map.get(1, 0)
    scale_pos_weight = float(benign_count) / float(attack_count) if attack_count > 0 else 1.0
    print(f"  Class ratio (Benign/Attack): {scale_pos_weight:.4f}")

    classifiers = get_classifiers(features_col, label_col, num_features, scale_pos_weight=scale_pos_weight)

    base_stages = [assembler, scaler]
    if extra_stages:
        base_stages.extend(extra_stages)

    results = OrderedDict()
    trained_models = OrderedDict()

    for name, clf in classifiers.items():
        print(f"\n{'─' * 60}")
        print(f"  Training: {name}")
        print(f"{'─' * 60}", flush=True)

        try:
            pipeline = Pipeline(stages=base_stages + [clf])
            model, preds, metrics = train_and_evaluate(
                pipeline, train_df, test_df, title=name
            )
            results[name] = metrics
            trained_models[name] = model
        except Exception as e:
            print(f"  [ERROR] Training {name}: {str(e)}")
            results[name] = {"accuracy": 0, "precision": 0, "recall": 0,
                             "f1": 0, "auc_roc": None, "auc_pr": None,
                             "training_time": 0, "error": str(e)}
            continue

    print(f"\n\n{'=' * 60}")
    print("  TRAINING HYBRID TOP-3 BAGGING ENSEMBLE (3-2-2 WEIGHTED)")
    print("=" * 60)
    
    base_results = {name: metrics for name, metrics in results.items() 
                    if "Bagging" not in name and "Ensemble" not in name}
    
    if len(base_results) < 3:
        print("  [WARN] Need at least 3 base models for Hybrid Bagging.")
        return results, trained_models

    sorted_names = sorted(base_results.keys(), key=lambda x: base_results[x].get("f1", 0), reverse=True)
    top_3 = sorted_names[:3]
    top_3_f1 = [base_results[name].get("f1", 0) for name in top_3]
    print(f"  Top-3 models: {', '.join(top_3)}")
    
    try:
        counts = [3, 2, 2]
        pipeline_dist = []
        total_weights = []
        
        for i, name in enumerate(top_3):
            clf = classifiers[name]
            pipeline = Pipeline(stages=base_stages + [clf])
            pipeline_dist.append((pipeline, counts[i]))
            
            for _ in range(counts[i]):
                total_weights.append(top_3_f1[i])
        
        start_time = time.time()
        ensemble_model = train_hybrid_bagging(
            pipeline_dist, train_df, balanced=False, feature_subset_rate=1.0
        )
        ensemble_model.weights = [w / sum(total_weights) for w in total_weights]
        
        training_time = time.time() - start_time
        
        start_pred = time.time()
        ens_preds = ensemble_model.transform(test_df)
        prediction_time = time.time() - start_pred
        
        ens_preds.cache().count()
        
        metrics = compute_metrics(ens_preds)
        metrics["training_time"] = training_time
        metrics["prediction_time"] = prediction_time
        
        total_size = sum([results[name].get("model_size_mb", 0.5) * count 
                         for name, count in zip(top_3, counts)])
        metrics["model_size_mb"] = total_size
        
        display_name = "Hybrid Bagging Ensemble (Top-3 Mixed + Weighted)"
        print_metrics(metrics, title=display_name)
        
        results[display_name] = metrics
        trained_models[display_name] = ensemble_model
        
        ens_preds.unpersist()
        
    except Exception as e:
        print(f"  [ERROR] Training Hybrid Bagging: {str(e)}")

    return results, trained_models


def ensemble_voting(
    trained_models: dict,
    test_df,
    results: dict = None,
    label_col: str = "label_binary",
    base_model_names: list = None,
    top_n: int = 3,
) -> dict:
    if base_model_names is None:
        if results is not None:
            base_results = {name: metrics for name, metrics in results.items()
                            if "Bagging" not in name and "Ensemble" not in name and "Voting" not in name}
            sorted_names = sorted(base_results.keys(), 
                                  key=lambda x: base_results[x].get("f1", 0), reverse=True)
            base_model_names = [n for n in sorted_names[:top_n] if n in trained_models]
        else:
            candidates = ["Random Forest", "GBT", "Logistic Regression",
                           "Decision Tree", "XGBoost"]
            base_model_names = [n for n in candidates if n in trained_models][:top_n]

    if len(base_model_names) < 2:
        print("  [WARN] Need at least 2 models for Ensemble Voting")
        return None

    print(f"\n{'─' * 60}")
    print(f"  Ensemble Voting (Top-{len(base_model_names)} by F1): {', '.join(base_model_names)}")
    if results is not None:
        for name in base_model_names:
            f1 = results.get(name, {}).get("f1", 0)
            print(f"    - {name}: F1 = {f1:.6f}")
    print(f"{'─' * 60}")

    start = time.time()

    test_with_id = test_df.withColumn("_row_id", monotonically_increasing_id())

    combined = None
    extract_prob_udf = udf(lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0, DoubleType())
    
    for i, name in enumerate(base_model_names):
        model = trained_models[name]
        preds = model.transform(test_with_id)
        
        if "probability" in preds.columns:
            preds = preds.select(
                "_row_id", label_col,
                extract_prob_udf(col("probability")).alias(f"prob_{i}")
            )
        else:
            preds = preds.select(
                "_row_id", label_col,
                col("prediction").alias(f"prob_{i}").cast(DoubleType())
            )
            
        if combined is None:
            combined = preds
        else:
            combined = combined.join(
                preds.select("_row_id", f"prob_{i}"), on="_row_id"
            )

    n = len(base_model_names)
    avg_prob_expr = sum([col(f"prob_{i}") for i in range(n)]) / float(n)
    
    vector_prob_udf = udf(lambda p: Vectors.dense([1.0 - p, p]), VectorUDT())
    
    combined = combined.withColumn("avg_probability", avg_prob_expr)
    combined = combined.withColumn("rawPrediction", col("avg_probability"))
    combined = combined.withColumn("probability", vector_prob_udf(col("avg_probability")))
    
    combined = combined.withColumn(
        "prediction",
        when(col("avg_probability") >= 0.5, 1.0).otherwise(0.0)
    )

    prediction_time = time.time() - start

    metrics = compute_metrics(combined, label_col=label_col)
    
    total_size = 0.0
    for name in base_model_names:
        m_size = get_model_size(trained_models[name])
        total_size += m_size

    metrics["training_time"] = 0.0
    metrics["prediction_time"] = prediction_time
    metrics["model_size_mb"] = total_size

    print_metrics(metrics, title=f"Ensemble Voting ({', '.join(base_model_names)})")

    return metrics


def plot_comparison(
    results: dict, title: str = "Algorithm Comparison",
    save_path: str = None, show: bool = True,
) -> None:
    names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, metric, color in zip(axes.flatten(), metric_names, colors):
        values = [results[n].get(metric, 0) for n in names]
        bars = ax.barh(names, values, color=color, alpha=0.85)
        ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
        ax.set_xlim(min(values) - 0.02 if min(values) > 0.02 else 0, 1.005)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_training_time(
    results: dict, title: str = "Training Time Comparison",
    save_path: str = None, show: bool = True,
) -> None:
    names = list(results.keys())
    times = [results[n].get("training_time", 0) for n in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, times, color=colors)
    for bar, val in zip(bars, times):
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}s", va="center", fontsize=9)
    plt.xlabel("Time (seconds)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_time(
    results: dict, title: str = "Prediction Time Comparison",
    save_path: str = None, show: bool = True,
) -> None:
    names = list(results.keys())
    times = [results[n].get("prediction_time", 0) for n in names]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, times, color=colors)
    for bar, val in zip(bars, times):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}s", va="center", fontsize=9)
    plt.xlabel("Time (seconds)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_size(
    results: dict, title: str = "Model Size Comparison",
    save_path: str = None, show: bool = True,
) -> None:
    names = list(results.keys())
    sizes = [results[n].get("model_size_mb", 0) for n in names]
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(names)))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, sizes, color=colors)
    for bar, val in zip(bars, sizes):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f} MB", va="center", fontsize=9)
    plt.xlabel("Size (MB)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices(
    results: dict, title: str = "Confusion Matrices",
    save_path: str = None, show: bool = True,
) -> None:
    valid = {k: v for k, v in results.items()
             if all(key in v for key in ["TP", "TN", "FP", "FN"]) and v.get("TP", 0) + v.get("TN", 0) > 0}
    if not valid:
        print("  [WARN] No confusion matrix data available.")
        return

    n = len(valid)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))

    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, metrics) in enumerate(valid.items()):
        ax = axes[idx]
        cm = np.array([[metrics["TN"], metrics["FP"]],
                        [metrics["FN"], metrics["TP"]]])

        cm_norm = cm.astype(float) / cm.sum() if cm.sum() > 0 else cm.astype(float)

        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Benign", "Attack"],
                    yticklabels=["Benign", "Attack"],
                    cbar=False, annot_kws={"fontsize": 10})
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        display_name = name if len(name) <= 25 else name[:22] + "..."
        ax.set_title(display_name, fontsize=10, fontweight="bold")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curves(
    trained_models: dict, test_df,
    label_col: str = "label_binary",
    title: str = "ROC Curves",
    save_path: str = None, show: bool = True,
    max_points: int = 200,
) -> None:
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import DoubleType

    extract_prob = udf(lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0, DoubleType())

    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(trained_models), 10)))

    for idx, (name, model) in enumerate(trained_models.items()):
        try:
            preds = model.transform(test_df)

            if "probability" not in preds.columns:
                print(f"  [WARN] Skipping {name}: no probability column")
                continue

            prob_df = preds.select(
                col(label_col).alias("label"),
                extract_prob(col("probability")).alias("prob_pos"),
            ).toPandas()

            labels = prob_df["label"].values
            probs = prob_df["prob_pos"].values

            thresholds = np.linspace(0, 1, max_points)
            tpr_list = []
            fpr_list = []

            total_pos = np.sum(labels == 1)
            total_neg = np.sum(labels == 0)

            if total_pos == 0 or total_neg == 0:
                continue

            for t in thresholds:
                predicted_pos = probs >= t
                tp = np.sum((predicted_pos) & (labels == 1))
                fp = np.sum((predicted_pos) & (labels == 0))
                tpr_list.append(tp / total_pos)
                fpr_list.append(fp / total_neg)

            sorted_pairs = sorted(zip(fpr_list, tpr_list))
            fpr_sorted = np.array([p[0] for p in sorted_pairs])
            tpr_sorted = np.array([p[1] for p in sorted_pairs])
            
            if hasattr(np, "trapezoid"):
                auc_val = np.trapezoid(tpr_sorted, fpr_sorted)
            elif hasattr(np, "trapz"):
                auc_val = np.trapz(tpr_sorted, fpr_sorted)
            else:
                auc_val = np.sum((fpr_sorted[1:] - fpr_sorted[:-1]) * (tpr_sorted[1:] + tpr_sorted[:-1]) / 2)

            color = colors[idx % len(colors)]
            display_name = name if len(name) <= 20 else name[:17] + "..."
            plt.plot(fpr_sorted, tpr_sorted, color=color, linewidth=1.5,
                     label=f"{display_name} (AUC={auc_val:.4f})")

        except Exception as e:
            print(f"  [WARN] Skipping {name}: {str(e)}")
            continue

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random (AUC=0.5)")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def print_summary_table(results: dict, title: str = "") -> None:
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    df = pd.DataFrame(results).T
    display_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", 
                    "training_time", "prediction_time", "model_size_mb"]
    available = [c for c in display_cols if c in df.columns]
    for c in available:
        if c not in ["training_time", "prediction_time", "model_size_mb"]:
            df[c] = df[c].apply(lambda x: f"{x:.6f}" if not pd.isna(x) and x is not None else "N/A")
        elif c == "model_size_mb":
            df[c] = df[c].apply(lambda x: f"{x:.3f} MB" if not pd.isna(x) and x is not None else "N/A")
        else:
            df[c] = df[c].apply(lambda x: f"{x:.3f}s" if not pd.isna(x) and x is not None else "N/A")
    print(df[available].to_string())
    print(f"{'=' * 100}")


from reporting.report_generator import (
    export_results_to_html,
    export_multi_section_report,
)


def shap_explain_model(
    spark_model, test_df, feature_cols: list, output_dir: str,
    sample_size: int = 1000, label_col: str = "label_binary",
) -> dict:
    import shap
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = {}
    
    print(f"\n{'─' * 60}")
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print(f"{'─' * 60}")
    
    print(f"  [1/5] Collecting {sample_size} test samples to Pandas...")
    
    select_cols = feature_cols + [label_col]
    sample_df = test_df.select(select_cols).limit(sample_size)
    pdf = sample_df.toPandas()
    
    X_test = pdf[feature_cols].values
    y_test = pdf[label_col].values
    
    print(f"        Collected: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"        Attack ratio: {y_test.mean():.2%}")
    
    print("  [2/5] Extracting XGBoost model from PipelineModel...")
    
    xgb_model = None
    for stage in spark_model.stages:
        stage_class = type(stage).__name__
        if "XGBoost" in stage_class or "XGB" in stage_class:
            xgb_model = stage
            break
    
    if xgb_model is None:
        print("  [ERROR] No XGBoost stage found in pipeline. SHAP requires XGBoost.")
        print("  Available stages:", [type(s).__name__ for s in spark_model.stages])
        return saved_plots
    
    booster = xgb_model.get_booster()
    print(f"        Extracted booster: {booster.num_boosted_rounds()} rounds")
    
    print("  [3/5] Computing SHAP values (TreeExplainer)...")
    
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(X_test)
    
    shap_values.feature_names = feature_cols
    
    print(f"        SHAP values computed: shape {shap_values.values.shape}")
    
    print("  [4/5] Generating SHAP plots...")
    
    plt.figure(figsize=(12, 10))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Summary Plot - Feature Impact on Attack Detection", fontsize=14, pad=20)
    plt.tight_layout()
    path_summary = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(path_summary, dpi=200, bbox_inches='tight')
    plt.close()
    saved_plots["summary_beeswarm"] = path_summary
    print(f"        [INFO] Saved: {path_summary}")
    
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("SHAP Global Feature Importance (Top-20)", fontsize=14, pad=20)
    plt.tight_layout()
    path_bar = os.path.join(output_dir, "shap_feature_importance_bar.png")
    plt.savefig(path_bar, dpi=200, bbox_inches='tight')
    plt.close()
    saved_plots["importance_bar"] = path_bar
    print(f"        [INFO] Saved: {path_bar}")
    
    attack_indices = np.where(y_test == 1)[0]
    if len(attack_indices) > 0:
        attack_idx = attack_indices[0]
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[attack_idx], max_display=15, show=False)
        plt.title(f"SHAP Waterfall - Why Sample #{attack_idx} Was Classified as Attack", 
                  fontsize=12, pad=20)
        plt.tight_layout()
        path_wf_attack = os.path.join(output_dir, "shap_waterfall_attack.png")
        plt.savefig(path_wf_attack, dpi=200, bbox_inches='tight')
        plt.close()
        saved_plots["waterfall_attack"] = path_wf_attack
        print(f"        [INFO] Saved: {path_wf_attack}")
    
    benign_indices = np.where(y_test == 0)[0]
    if len(benign_indices) > 0:
        benign_idx = benign_indices[0]
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[benign_idx], max_display=15, show=False)
        plt.title(f"SHAP Waterfall - Why Sample #{benign_idx} Was Classified as Benign", 
                  fontsize=12, pad=20)
        plt.tight_layout()
        path_wf_benign = os.path.join(output_dir, "shap_waterfall_benign.png")
        plt.savefig(path_wf_benign, dpi=200, bbox_inches='tight')
        plt.close()
        saved_plots["waterfall_benign"] = path_wf_benign
        print(f"        [INFO] Saved: {path_wf_benign}")
    
    print("  [5/5] Generating SHAP feature ranking table...")
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_SHAP_Value": mean_abs_shap
    }).sort_values("Mean_SHAP_Value", ascending=False)
    
    csv_path = os.path.join(output_dir, "shap_feature_importance.csv")
    feature_importance.to_csv(csv_path, index=False)
    saved_plots["importance_csv"] = csv_path
    
    print(f"\n  {'─' * 50}")
    print(f"  Top-20 Features by Mean |SHAP Value|")
    print(f"  {'─' * 50}")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    print(f"  {'─' * 50}")
    for rank, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        print(f"  {rank:<6} {row['Feature']:<35} {row['Mean_SHAP_Value']:>12.6f}")
    
    print(f"\n  SHAP analysis completed! {len(saved_plots)} outputs saved to: {output_dir}")
    
    return saved_plots
