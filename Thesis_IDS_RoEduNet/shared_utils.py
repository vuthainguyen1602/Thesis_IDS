#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
SHARED UTILITIES - IDS (RoEduNet-SIMARGL2021 Dataset)
================================================================================
Contains: Spark configuration, data loading, preprocessing, evaluation functions,
and all classifier definitions.
Adapted from the CICIDS2017 project for cross-dataset validation.
================================================================================
"""

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# --- Python & Java Configuration ---
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[4] pyspark-shell'

# --- PySpark Imports ---
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, monotonically_increasing_id, udf
from pyspark.sql.types import StringType, DoubleType

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

# --- XGBoost Import (optional) ---
try:
    from xgboost.spark import SparkXGBClassifier
    HAS_XGBOOST = True
    print("XGBoost available")
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available (pip install xgboost)")

# --- LightGBM Import (optional — requires Java JAR in Spark) ---
try:
    from synapse.ml.lightgbm import LightGBMClassifier
    # Test if Java backend is actually available
    _test = LightGBMClassifier()
    del _test
    HAS_LIGHTGBM = True
    print("LightGBM available")
except (ImportError, TypeError, Exception):
    HAS_LIGHTGBM = False
    print("LightGBM not available (missing Java backend or pip install synapseml)")


# ==============================================================================
# SPARK SESSION INITIALIZATION
# ==============================================================================

def create_spark_session(app_name="IDS_Binary_Prediction"):
    """Create and return a SparkSession with optimized memory for ensembles."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # Use all available cores
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "8g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "100s")
        .config("spark.sql.shuffle.partitions", "16") # Reduce overhead for local mode
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark Version: {spark.version}")
    print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
    return spark


# ==============================================================================
# DATA PREPROCESSING FUNCTIONS
# ==============================================================================

def clean_column_names(df):
    """Normalize column names: lowercase, snake_case, remove special characters."""
    for col_name in df.columns:
        new_name = (
            col_name.strip()
            .lower()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        while "__" in new_name:
            new_name = new_name.replace("__", "_")
        df = df.withColumnRenamed(col_name, new_name)
    return df


def handle_infinity_values(df):
    """Replace Infinity and NaN values with null."""
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


def align_schema(df, ref_columns):
    """Add missing columns to unify schema."""
    for c in ref_columns:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(StringType()))
    return df.select(ref_columns)


# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

def load_and_prepare_data(spark):
    """
    Load preprocessed RoEduNet-SIMARGL2021 data from saved parquet files.
    Run data_preparation.py first if parquet files don't exist.
    Returns: (df, train_df, test_df, feature_cols)
    """
    output_dir = "/Users/thainguyenvu/Desktop/Thesis_IDS/Thesis_IDS_RoEduNet/data"
    train_path = os.path.join(output_dir, "train_data.parquet")
    test_path = os.path.join(output_dir, "test_data.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Parquet data not found. Run data_preparation.py first!\n"
            f"  Expected: {train_path}\n"
            f"  Expected: {test_path}"
        )

    print("=" * 60)
    print("LOADING RoEduNet-SIMARGL2021 DATA FROM PARQUET")
    print("=" * 60)

    train_df = spark.read.parquet(train_path)
    test_df = spark.read.parquet(test_path)

    train_df = train_df.cache()
    test_df = test_df.cache()

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"  Training set: {train_count:,} samples")
    print(f"  Test set:     {test_count:,} samples")

    # Combine for full df reference
    df = train_df.unionByName(test_df).cache()

    # Identify numeric feature columns
    # Exclude: IP addresses, timestamps, protocol, tag, labels
    exclude_cols = ["label", "label_binary", "src_ip", "dst_ip",
                    "time_first", "time_last", "proto", "tag"]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and dict(df.dtypes)[c] in ["double", "float", "int", "bigint", "long"]
    ]
    print(f"  Number of numeric features: {len(feature_cols)}")
    print("=" * 60)

    return df, train_df, test_df, feature_cols


# ==============================================================================
# CLASSIFIER DEFINITIONS
# ==============================================================================

def get_classifiers(features_col, label_col="label_binary", num_features=50, scale_pos_weight=None):
    """
    Return an OrderedDict containing all classification algorithms.
    
    Parameters:
        features_col: name of the feature vector column
        label_col: name of the label column
        num_features: number of features (needed for MLP layer specification)
    
    Returns:
        OrderedDict { algorithm_name: classifier_instance }
    """
    classifiers = OrderedDict()

    # ──────────────────────────────────────────────────────────────────────
    # Hyperparameters optimized for BEST ACCURACY on CICIDS2017
    # References: Q1 2024-2025 research papers
    #   - RF: n_estimators=200, max_depth=15 → 98.56%+ (Mesopotamian Press 2024)
    #   - XGBoost: n_est=300, max_depth=8, lr=0.05 → ~99.9% (QuestJournals 2024)
    #   - LightGBM: n_leaves=63, n_iter=300, lr=0.05 → 99.98% (BAS.bg 2024)
    #   - MLP: [128, 64, 32] hidden → 99%+ (ResearchGate 2025)
    #   - GBT: maxIter=150, maxDepth=6, stepSize=0.05 (arXiv 2024)
    #   - DT-PCA: maxDepth=15 → 99%+ (ResearchGate 2025)
    # ──────────────────────────────────────────────────────────────────────

    # 1. Decision Tree
    classifiers["Decision Tree"] = DecisionTreeClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        maxDepth=15,               # Deeper trees capture more patterns (2025 DT-PCA study)
        minInstancesPerNode=5,     # Smaller leaves → finer granularity
        impurity="entropy",        # Entropy outperforms gini on CICIDS2017 (ICCS 2024)
        seed=42,
    )

    # 2. Logistic Regression
    classifiers["Logistic Regression"] = LogisticRegression(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=200,               # More iterations for full convergence
        regParam=0.001,            # Lighter regularization → better fit on large data
        elasticNetParam=0.0,       # Pure L2 (Ridge) — more stable for IDS features
        family="binomial",
        threshold=0.4,             # Slightly lower threshold to catch more attacks
    )

    # 3. Support Vector Machine (LinearSVC)
    classifiers["SVM (LinearSVC)"] = LinearSVC(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=100,               # More iterations for better convergence
        regParam=0.001,            # Lighter regularization for better decision boundary
        threshold=0.0,
    )

    # 4. Naive Bayes (Gaussian - supports negative feature values)
    classifiers["Naive Bayes"] = NaiveBayes(
        featuresCol=features_col,
        labelCol=label_col,
        modelType="gaussian",
        smoothing=1.0,             # Laplace smoothing
    )

    # 5. Random Forest — 200 trees, deeper (2024 research: 98.56%+ accuracy)
    classifiers["Random Forest"] = RandomForestClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        numTrees=200,              # More trees → better stability & accuracy
        maxDepth=15,               # Deeper → captures complex attack patterns
        minInstancesPerNode=5,     # Finer splits for higher recall
        featureSubsetStrategy="sqrt",
        subsamplingRate=1.0,       # Full data for each tree (max accuracy)
        seed=42,
    )

    # 6. Gradient Boosted Trees — 150 rounds, deeper, lower LR
    classifiers["GBT"] = GBTClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=150,               # More boosting rounds for thorough learning
        maxDepth=6,                # Slightly deeper trees for better capture
        stepSize=0.05,             # Lower LR → more precise convergence
        subsamplingRate=0.8,       # Stochastic gradient boosting
        seed=42,
    )

    # 7. XGBoost — 300 estimators, deeper trees (2024: ~99.9% accuracy)
    if HAS_XGBOOST:
        xgb_params = dict(
            features_col=features_col,
            label_col=label_col,
            num_workers=4,
            n_estimators=300,        # More trees with lower LR = higher accuracy
            max_depth=8,             # Deeper trees to capture complex patterns
            learning_rate=0.05,      # Lower LR → more precise, compensated by more trees
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,      # Smaller → finer splits, better recall
            gamma=0.05,              # Lower threshold → allow more splits
            reg_alpha=0.5,           # Moderate L1 regularization
            reg_lambda=1.0,          # L2 regularization
            use_gpu=False,
        )
        if scale_pos_weight is not None:
            xgb_params["scale_pos_weight"] = scale_pos_weight
        classifiers["XGBoost"] = SparkXGBClassifier(**xgb_params)

    # 8. LightGBM — 300 iterations, 63 leaves (2024: 99.98% accuracy)
    if HAS_LIGHTGBM:
        classifiers["LightGBM"] = LightGBMClassifier(
            featuresCol=features_col,
            labelCol=label_col,
            numIterations=300,       # More rounds for thorough learning
            numLeaves=63,            # 2^6-1: richer tree structure
            maxDepth=8,              # Deeper for complex pattern capture
            learningRate=0.05,       # Lower LR → more precise convergence
            minDataInLeaf=20,        # Moderate leaf size → good generalization
            featureFraction=0.8,
            baggingFraction=0.8,
            baggingFreq=5,
            lambdaL1=0.5,           # L1 regularization (prevents overfitting)
            lambdaL2=1.0,           # L2 regularization
            objective="binary",
        )

    # 9. Multi-Layer Perceptron — deeper architecture (2025: 99%+ accuracy)
    #    Architecture: input → 128 → 64 → 32 → output
    layers = [num_features, 128, 64, 32, 2]
    classifiers["MLP"] = MultilayerPerceptronClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        layers=layers,
        maxIter=200,               # More iterations for deep network convergence
        blockSize=128,             # Smaller batch → more updates per epoch
        stepSize=0.01,             # Lower LR for stable training of deeper network
        seed=42,
    )

    return classifiers


# ==============================================================================
# EVALUATION AND UTILITY FUNCTIONS
# ==============================================================================

def compute_metrics(predictions, label_col="label_binary"):
    """
    Compute evaluation metrics for binary classification in an optimized way.
    Uses grouped counts to avoid multiple Spark jobs.
    """
    # 1. Confusion Matrix (Optimized to a single grouped count)
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

    # 2. AUC-ROC & AUC-PR (Requires rawPrediction)
    auc_roc = None
    auc_pr = None
    if "rawPrediction" in predictions.columns:
        try:
            evaluator_roc = BinaryClassificationEvaluator(
                labelCol=label_col, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            auc_roc = evaluator_roc.evaluate(predictions)

            evaluator_pr = BinaryClassificationEvaluator(
                labelCol=label_col, rawPredictionCol="rawPrediction",
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


def print_metrics(metrics, title=""):
    """Print evaluation results in a formatted layout."""
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


def get_model_size(model):
    """
    Estimate Spark ML model size by saving it to a temporary path.
    Returns size in Megabytes (MB).
    """
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
        
        return total_size / (1024 * 1024) # Bytes to MB
    except Exception:
        return 0.0
    finally:
        if os.path.exists(temp_path):
            try:
                shutil.rmtree(temp_path)
            except Exception:
                pass


def train_and_evaluate(pipeline, train_df, test_df, title=""):
    """Train a pipeline, predict on test set, and evaluate."""
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


# ==============================================================================
# BAGGING ENSEMBLE LEARNING (MANUAL IMPLEMENTATION)
# ==============================================================================

class BaggingModel:
    """A wrapper for multiple models trained on bootstrap samples."""
    def __init__(self, models, names=None, weights=None):
        self.models = models
        self.names = names or [f"Model_{i}" for i in range(len(models))]
        # Normalize weights if provided
        if weights is not None:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

    def transform(self, df):
        """
        Produce predictions using Soft Voting (Probability Averaging).
        """
        from pyspark.sql.functions import col, when, monotonically_increasing_id, udf
        from pyspark.sql.types import DoubleType

        # Add row_id to align predictions
        # IMPORTANT: We MUST cache this to ensure monotonically_increasing_id is stable
        df_with_id = df.withColumn("_row_id", monotonically_increasing_id()).cache()
        df_with_id.count()
        
        # UDF to extract probability of positive class
        extract_prob_udf = udf(lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0, DoubleType())
        
        # To avoid a deep join chain, we'll collect predictions in a list and join once if possible
        # or join iteratively but cache intermediate results if the chain gets too long.
        combined_result = df_with_id.select("_row_id", "label_binary")
        
        for i, model in enumerate(self.models):
            preds = model.transform(df_with_id)
            preds = preds.withColumn(f"prob_{i}", extract_prob_udf(col("probability")))
            combined_result = combined_result.join(preds.select("_row_id", f"prob_{i}"), on="_row_id")
            
            # For every 5 models, we checkpoint the join to break the lineage and prevent StackOverflow
            if (i + 1) % 5 == 0:
                combined_result = combined_result.localCheckpoint()

        # Soft Voting: Weighted probability across all models
        weighted_probs = []
        for i in range(len(self.models)):
            weighted_probs.append(col(f"prob_{i}") * self.weights[i])
        
        avg_prob_expr = sum(weighted_probs)
        
        final_res = combined_result.withColumn("avg_probability", avg_prob_expr)
        final_res = final_res.withColumn(
            "prediction", 
            when(col("avg_probability") >= 0.5, 1.0).otherwise(0.0)
        )
        
        # Note: We don't unpersist df_with_id here because final_res depends on it.
        # User should unpersist the final returned dataframe after counting/consuming.
        return final_res

    def save(self, path):
        # Placeholder for save logic (complex for custom objects in Spark)
        pass

def train_manual_bagging(base_pipeline, train_df, num_models=5, label_col="label_binary", benign_ratio=0.7):
    """
    Train a Balanced Bagging ensemble using Manual Bootstrap Sampling.
    Ratio: benign_ratio (Benign) / (1 - benign_ratio) (Attack).
    Example: benign_ratio=0.7 -> 70% Benign, 30% Attack.
    """
    models = []
    print(f"  Training Balanced Bagging Ensemble ({num_models} models, {benign_ratio*100:.0f}/{ (1-benign_ratio)*100:.0f} ratio)...")
    
    # Split data by class
    attack_df = train_df.filter(col(label_col) == 1).cache()
    benign_df = train_df.filter(col(label_col) == 0).cache()
    
    attack_count = attack_df.count()
    benign_count = benign_df.count()
    
    if attack_count == 0 or benign_count == 0:
        return None

    # Calculate count for benign based on fixed attack count to reach target ratio
    # target_ratio = target_benign_count / (target_benign_count + attack_count)
    # target_benign_count = (benign_ratio * attack_count) / (1 - benign_ratio)
    target_benign_count = int((benign_ratio * attack_count) / (1 - benign_ratio))
    benign_fraction = float(target_benign_count) / benign_count
    
    for i in range(num_models):
        print(f"   - Building bootstrap model {i+1}/{num_models}...")
        
        # 1. Bootstrap sampling for Attack class
        attack_sample = attack_df.sample(withReplacement=True, fraction=1.0, seed=42 + i)
        
        # 2. Balanced sampling for Benign class
        benign_sample = benign_df.sample(withReplacement=True, fraction=min(benign_fraction, 10.0), seed=100 + i)
        
        # 3. Combine
        balanced_bag = attack_sample.unionAll(benign_sample)
        
        # 4. Train
        model = base_pipeline.fit(balanced_bag)
        models.append(model)
        
    # Cleanup training cache
    attack_df.unpersist()
    benign_df.unpersist()
    
    return BaggingModel(models)

def train_hybrid_bagging(pipeline_distribution, train_df, label_col="label_binary", 
                         benign_ratio=0.7, balanced=True, feature_list=None, feature_subset_rate=1.0):
    """
    Train a hybrid ensemble where different models can have different replica counts.
    
    Parameters:
        pipeline_distribution: list of (Pipeline, num_replicas)
        train_df: training data
        label_col: target column
        benign_ratio: ratio of benign samples in each bag (default 0.7)
        balanced: whether to use class-aware sampling (default True)
        feature_list: list of all possible features (required for feature bagging)
        feature_subset_rate: fraction of features to pick per model (default 1.0)
    """
    total_models = sum([count for _, count in pipeline_distribution])
    models = []
    
    if balanced:
        # Separate classes for class-aware sampling
        attack_df = train_df.filter(col(label_col) == 1.0).cache()
        benign_df = train_df.filter(col(label_col) == 0.0).cache()
        
        attack_count = attack_df.count()
        benign_count = benign_df.count()
        
        target_benign_count = int((benign_ratio * attack_count) / (1 - benign_ratio))
        benign_fraction = float(target_benign_count) / benign_count
    else:
        # Cache full training set for performance
        full_train_df = train_df.cache()

    current_idx = 0
    for base_pipeline, num_replicas in pipeline_distribution:
        for i in range(num_replicas):
            current_idx += 1
            print(f"   - Building hybrid model {current_idx}/{total_models} (Balanced={balanced})...")
            
            if balanced:
                # Bootstrap sampling per class
                attack_sample = attack_df.sample(withReplacement=True, fraction=1.0, seed=42 + current_idx)
                benign_sample = benign_df.sample(withReplacement=True, fraction=min(benign_fraction, 10.0), seed=100 + current_idx)
                bag_df = attack_sample.unionAll(benign_sample)
            else:
                # Standard Bootstrap sampling on full dataset
                bag_df = full_train_df.sample(withReplacement=True, fraction=1.0, seed=42 + current_idx)
            
            # --- Feature Bagging: Random Subset of Features ---
            # We copy the pipeline to avoid side-effects on the original template
            bag_pipeline = base_pipeline.copy()
            
            if feature_list and feature_subset_rate < 1.0:
                k = max(1, int(len(feature_list) * feature_subset_rate))
                # Use current_idx as part of seed for reproducibility
                random.seed(42 + current_idx)
                selected_features = random.sample(feature_list, k)
                print(f"     * Feature Bagging: {k}/{len(feature_list)} random features selected")
                
                # Update the VectorAssembler stage
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


# ==============================================================================
# CLASSIFICATION FRAMEWORK
# ==============================================================================

def run_all_classifiers(assembler, scaler, train_df, test_df,
                        features_col, num_features, label_col="label_binary",
                        extra_stages=None):
    """
    Run all classification algorithms and collect results.
    
    Parameters:
        assembler: configured VectorAssembler
        scaler: configured StandardScaler
        train_df, test_df: training/test data
        features_col: name of the feature column input to classifiers
        num_features: dimensionality of the feature vector input to classifiers
        label_col: name of the label column
        extra_stages: additional pipeline stages (e.g., PCA) inserted between scaler and classifier
    
    Returns:
        (results_dict, trained_models_dict)
    """
    # Calculate scale_pos_weight for imbalanced data (benign_count / attack_count)
    benign_count = train_df.filter(col(label_col) == 0).count()
    attack_count = train_df.filter(col(label_col) == 1).count()
    scale_pos_weight = float(benign_count) / float(attack_count) if attack_count > 0 else 1.0
    print(f"  Class ratio (Benign/Attack): {scale_pos_weight:.4f}")

    classifiers = get_classifiers(features_col, label_col, num_features, scale_pos_weight=scale_pos_weight)

    base_stages = [assembler, scaler]
    if extra_stages:
        base_stages.extend(extra_stages)

    results = OrderedDict()
    trained_models = OrderedDict()

    # Regular Classifiers
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
            print(f"  ERROR training {name}: {str(e)}")
            results[name] = {"accuracy": 0, "precision": 0, "recall": 0,
                             "f1": 0, "auc_roc": None, "auc_pr": None,
                             "training_time": 0, "error": str(e)}
            continue

    # --- ADDED: Hybrid Bagging Ensemble (Top-3 Mixed + Weighted) ---
    print(f"\n\n{'█' * 60}")
    print("  TRAINING HYBRID TOP-3 BAGGING ENSEMBLE (3-2-2 WEIGHTED)")
    print("█" * 60)
    
    base_results = {name: metrics for name, metrics in results.items() 
                    if "Bagging" not in name and "Ensemble" not in name}
    
    if len(base_results) < 3:
        print("  Need at least 3 base models for Hybrid Bagging.")
        return results, trained_models

    # Get Top-3 names by F1-score
    sorted_names = sorted(base_results.keys(), key=lambda x: base_results[x].get("f1", 0), reverse=True)
    top_3 = sorted_names[:3]
    top_3_f1 = [base_results[name].get("f1", 0) for name in top_3]
    print(f"  Top-3 models: {', '.join(top_3)}")
    
    try:
        # Configuration: 3x Rank 1, 2x Rank 2, 2x Rank 3 (7 models total → faster)
        counts = [3, 2, 2]
        pipeline_dist = []
        total_weights = []
        
        for i, name in enumerate(top_3):
            clf = classifiers[name]
            pipeline = Pipeline(stages=base_stages + [clf])
            pipeline_dist.append((pipeline, counts[i]))
            
            # Weight is the F1-score. Repeat for each replica.
            # (Weighting replicas equally within the same model type)
            for _ in range(counts[i]):
                total_weights.append(top_3_f1[i])
        
        start_time = time.time()
        # Train Hybrid Bagging (Standard bootstrap)
        ensemble_model = train_hybrid_bagging(
            pipeline_dist, train_df, balanced=False, feature_subset_rate=1.0
        )
        # Apply weights to the ensemble
        ensemble_model.weights = [w / sum(total_weights) for w in total_weights]
        
        training_time = time.time() - start_time
        
        start_pred = time.time()
        ens_preds = ensemble_model.transform(test_df)
        prediction_time = time.time() - start_pred
        
        ens_preds.cache().count()
        
        metrics = compute_metrics(ens_preds)
        metrics["training_time"] = training_time
        metrics["prediction_time"] = prediction_time
        
        # Estimate size
        total_size = sum([results[name].get("model_size_mb", 0.5) * count 
                         for name, count in zip(top_3, counts)])
        metrics["model_size_mb"] = total_size
        
        display_name = "Hybrid Bagging Ensemble (Top-3 Mixed + Weighted)"
        print_metrics(metrics, title=display_name)
        
        results[display_name] = metrics
        trained_models[display_name] = ensemble_model
        
        ens_preds.unpersist()
        
    except Exception as e:
        print(f"  ERROR training Hybrid Bagging: {str(e)}")

    return results, trained_models


# ==============================================================================
# ENSEMBLE LEARNING - MAJORITY VOTING
# ==============================================================================

def ensemble_voting(trained_models, test_df, results=None, label_col="label_binary",
                    base_model_names=None, top_n=3):
    """
    Create an Ensemble using Majority Voting.
    Selects Top-N models by F1-Score for voting.
    
    Parameters:
        trained_models: dict {name: fitted_model}
        test_df: test data
        results: dict {name: metrics_dict} containing F1 scores for model selection
        label_col: name of the label column
        base_model_names: explicit list of model names (overrides auto-selection)
        top_n: number of top models to select (default: 3)
    
    Returns:
        dict containing ensemble metrics
    """
    if base_model_names is None:
        if results is not None:
            # Select Top-N models by F1-Score (exclude ensemble/bagging models)
            base_results = {name: metrics for name, metrics in results.items()
                            if "Bagging" not in name and "Ensemble" not in name and "Voting" not in name}
            sorted_names = sorted(base_results.keys(), 
                                  key=lambda x: base_results[x].get("f1", 0), reverse=True)
            base_model_names = [n for n in sorted_names[:top_n] if n in trained_models]
        else:
            # Fallback: select from priority list
            candidates = ["Random Forest", "GBT", "Logistic Regression",
                           "Decision Tree", "XGBoost"]
            base_model_names = [n for n in candidates if n in trained_models][:top_n]

    if len(base_model_names) < 2:
        print("  Need at least 2 models for Ensemble Voting")
        return None

    print(f"\n{'─' * 60}")
    print(f"  Ensemble Voting (Top-{len(base_model_names)} by F1): {', '.join(base_model_names)}")
    if results is not None:
        for name in base_model_names:
            f1 = results.get(name, {}).get("f1", 0)
            print(f"    - {name}: F1 = {f1:.6f}")
    print(f"{'─' * 60}")

    start = time.time()

    # Add row_id for joining
    test_with_id = test_df.withColumn("_row_id", monotonically_increasing_id())

    # Collect predictions from each model
    combined = None
    for i, name in enumerate(base_model_names):
        model = trained_models[name]
        preds = model.transform(test_with_id)
        preds = preds.select(
            "_row_id", label_col,
            col("prediction").alias(f"pred_{i}").cast(DoubleType())
        )
        if combined is None:
            combined = preds
        else:
            combined = combined.join(
                preds.select("_row_id", f"pred_{i}"), on="_row_id"
            )

    # Majority voting: sum of predictions > n/2 -> 1, otherwise -> 0
    n = len(base_model_names)
    vote_expr = sum([col(f"pred_{i}") for i in range(n)])
    combined = combined.withColumn(
        "prediction",
        when(vote_expr > n / 2.0, 1.0).otherwise(0.0)
    )

    # Voting time (this is inference time)
    prediction_time = time.time() - start

    # Compute metrics using optimized function
    metrics = compute_metrics(combined, label_col=label_col)
    
    # Sum sizes of base models
    total_size = 0.0
    for name in base_model_names:
        m_size = get_model_size(trained_models[name])
        total_size += m_size

    metrics["training_time"] = 0.0 # Voting has no training
    metrics["prediction_time"] = prediction_time
    metrics["model_size_mb"] = total_size

    print_metrics(metrics, title=f"Ensemble Voting ({', '.join(base_model_names)})")

    return metrics


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_comparison(results, title="Algorithm Comparison", save_path=None, show=True):
    """
    Plot bar charts comparing Accuracy, Precision, Recall, F1 for all algorithms.
    """
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

def plot_training_time(results, title="Training Time Comparison", save_path=None, show=True):
    """Plot horizontal bar chart comparing training times."""
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


def plot_prediction_time(results, title="Prediction Time Comparison", save_path=None, show=True):
    """Plot horizontal bar chart comparing prediction times."""
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


def plot_model_size(results, title="Model Size Comparison", save_path=None, show=True):
    """Plot horizontal bar chart comparing model sizes."""
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


def print_summary_table(results, title=""):
    """Print a formatted summary table of results."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    df = pd.DataFrame(results).T
    display_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", 
                    "training_time", "prediction_time", "model_size_mb"]
    available = [c for c in display_cols if c in df.columns]
    # Format numbers
    for c in available:
        if c not in ["training_time", "prediction_time", "model_size_mb"]:
            df[c] = df[c].apply(lambda x: f"{x:.6f}" if x is not None else "N/A")
        elif c == "model_size_mb":
            df[c] = df[c].apply(lambda x: f"{x:.3f} MB" if x is not None else "N/A")
        else:
            df[c] = df[c].apply(lambda x: f"{x:.3f}s" if x is not None else "N/A")
    print(df[available].to_string())
    print(f"{'=' * 100}")


def export_results_to_html(results, title="Experiment Results", output_path="report.html", chart_paths=None):
    """
    Export results dictionary to a styled HTML report.
    chart_paths: Optional list of paths to .png files to embed
    """
    import base64
    
    df = pd.DataFrame(results).T
    display_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", 
                    "training_time", "prediction_time", "model_size_mb"]
    available = [c for c in display_cols if c in df.columns]
    
    # Header & CSS
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f7f9; color: #333; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
            h2 {{ color: #5f6368; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; color: #202124; font-weight: 600; }}
            tr:hover {{ background-color: #f1f3f4; }}
            .metric {{ font-weight: bold; color: #1a73e8; }}
            .chart-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 30px; }}
            .chart-box {{ flex: 1; min-width: 450px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; }}
            .chart-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #70757a; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Algorithm</th>
                        {" ".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in available)}
                    </tr>
                </thead>
                <tbody>
    """
    
    for name, row in df.iterrows():
        html += f"<tr><td><strong>{name}</strong></td>"
        for col_name in available:
            val = row.get(col_name)
            if pd.isna(val) or val is None:
                formatted = "N/A"
            elif col_name in ["training_time", "prediction_time"]:
                formatted = f"{val:.3f}s"
            elif col_name == "model_size_mb":
                formatted = f"{val:.3f} MB"
            else:
                formatted = f"{val:.6f}"
            html += f"<td>{formatted}</td>"
        html += "</tr>"
        
    html += """
                </tbody>
            </table>
    """
    
    # Embedded Charts
    if chart_paths:
        html += '<h2>Visualizations</h2><div class="chart-container">'
        for cp in chart_paths:
            if os.path.exists(cp):
                try:
                    with open(cp, "rb") as f:
                        data = base64.b64encode(f.read()).decode()
                        filename = os.path.basename(cp)
                        html += f"""
                        <div class="chart-box">
                            <h3>{filename}</h3>
                            <img src="data:image/png;base64,{data}" alt="{filename}">
                        </div>
                        """
                except Exception as e:
                    html += f"<p>Error embedding {cp}: {str(e)}</p>"
        html += '</div>'
        
    html += f"""
            <div class="footer">
                <p>IDS Thesis Binary Prediction - Apache Spark Result Report</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✓ Report exported to: {output_path}")


def export_multi_section_report(sections, title="Experiment Results", output_path="report.html"):
    """
    Export multiple experiment groups into a single HTML report.
    sections: List of dicts, each with:
        "section_title": str
        "results": dict (classifier metrics)
        "chart_paths": list (optional)
    """
    import base64
    
    # CSS remains same as original but we apply for multiple sections
    css = """
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f7f9; color: #333; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; text-align: center; margin-bottom: 40px; }
        .section { margin-bottom: 60px; padding-top: 20px; border-top: 1px solid #eee; }
        h2 { color: #1a73e8; margin-top: 0; background: #e8f0fe; padding: 10px 20px; border-radius: 4px; }
        h3 { color: #5f6368; margin-top: 25px; border-left: 4px solid #1a73e8; padding-left: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; color: #202124; font-weight: 600; }
        tr:hover { background-color: #f1f3f4; }
        .chart-container { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 25px; }
        .chart-box { flex: 1; min-width: 450px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; }
        .chart-box img { max-width: 100%; height: auto; border-radius: 4px; }
        .footer { margin-top: 50px; font-size: 12px; color: #70757a; text-align: center; }
    """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>{css}</style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p style="text-align:center;">Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    """

    for sec in sections:
        sec_title = sec.get("section_title", "Section")
        results = sec.get("results", {})
        chart_paths = sec.get("chart_paths", [])
        
        df = pd.DataFrame(results).T
        display_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", 
                        "training_time", "prediction_time", "model_size_mb"]
        available = [c for c in display_cols if c in df.columns]

        html += f"""
            <div class="section">
                <h2>{sec_title}</h2>
                <h3>Summary Results</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            {" ".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in available)}
                        </tr>
                    </thead>
                    <tbody>
        """

        for name, row in df.iterrows():
            html += f"<tr><td><strong>{name}</strong></td>"
            for col_name in available:
                val = row.get(col_name)
                if pd.isna(val) or val is None:
                    formatted = "N/A"
                elif col_name in ["training_time", "prediction_time"]:
                    formatted = f"{val:.3f}s"
                elif col_name == "model_size_mb":
                    formatted = f"{val:.3f} MB"
                else:
                    formatted = f"{val:.6f}"
                html += f"<td>{formatted}</td>"
            html += "</tr>"

        html += "</tbody></table>"

        if chart_paths:
            html += '<div class="chart-container">'
            for cp in chart_paths:
                if os.path.exists(cp):
                    try:
                        with open(cp, "rb") as f:
                            data = base64.b64encode(f.read()).decode()
                            filename = os.path.basename(cp)
                            html += f"""
                            <div class="chart-box">
                                <p style="font-weight:600;margin:5px 0;">{filename}</p>
                                <img src="data:image/png;base64,{data}" alt="{filename}">
                            </div>
                            """
                    except Exception as e:
                        html += f"<p>Error embedding {cp}: {str(e)}</p>"
            html += '</div>'
        
        html += "</div>" # close section

    html += f"""
            <div class="footer">
                <p>IDS Thesis Binary Prediction - Comprehensive Experiment Report</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✓ Comprehensive Report exported to: {output_path}")


# ==============================================================================
# SHAP EXPLAINABILITY (XAI)
# ==============================================================================

def shap_explain_model(spark_model, test_df, feature_cols, output_dir,
                       sample_size=1000, label_col="label_binary"):
    """
    Generate SHAP explainability plots for a trained Spark ML PipelineModel.
    
    This function extracts the XGBoost booster from the trained pipeline,
    collects a sample of test data to Pandas, and computes SHAP values
    using TreeExplainer for fast, exact computation.
    
    Parameters:
        spark_model: Trained PipelineModel (must contain XGBoost stage)
        test_df: PySpark test DataFrame
        feature_cols: list of original feature column names
        output_dir: directory to save plots
        sample_size: number of test samples for SHAP computation
        label_col: name of the label column
    
    Returns:
        dict with paths to saved plots
    """
    import shap
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = {}
    
    print(f"\n{'─' * 60}")
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print(f"{'─' * 60}")
    
    # ── Step 1: Collect sample test data to Pandas ──
    print(f"  [1/5] Collecting {sample_size} test samples to Pandas...")
    
    # Select only feature columns + label
    select_cols = feature_cols + [label_col]
    sample_df = test_df.select(select_cols).limit(sample_size)
    pdf = sample_df.toPandas()
    
    X_test = pdf[feature_cols].values
    y_test = pdf[label_col].values
    
    print(f"        Collected: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"        Attack ratio: {y_test.mean():.2%}")
    
    # ── Step 2: Extract XGBoost booster from Spark Pipeline ──
    print("  [2/5] Extracting XGBoost model from PipelineModel...")
    
    xgb_model = None
    for stage in spark_model.stages:
        stage_class = type(stage).__name__
        # Look for the XGBoost model stage (handles SparkXGBClassifierModel, XGBoostClassificationModel, etc.)
        if "XGBoost" in stage_class or "XGB" in stage_class:
            xgb_model = stage
            break
    
    if xgb_model is None:
        print("  ERROR: No XGBoost stage found in pipeline. SHAP requires XGBoost.")
        print("  Available stages:", [type(s).__name__ for s in spark_model.stages])
        return saved_plots
    
    # Extract the native booster
    booster = xgb_model.get_booster()
    print(f"        Extracted booster: {booster.num_boosted_rounds()} rounds")
    
    # ── Step 3: Compute SHAP values ──
    print("  [3/5] Computing SHAP values (TreeExplainer)...")
    
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(X_test)
    
    # Assign feature names
    shap_values.feature_names = feature_cols
    
    print(f"        SHAP values computed: shape {shap_values.values.shape}")
    
    # ── Step 4: Generate Plots ──
    print("  [4/5] Generating SHAP plots...")
    
    # --- Plot 1: Summary Plot (Beeswarm) ---
    plt.figure(figsize=(12, 10))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Summary Plot - Feature Impact on Attack Detection", fontsize=14, pad=20)
    plt.tight_layout()
    path_summary = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(path_summary, dpi=200, bbox_inches='tight')
    plt.close()
    saved_plots["summary_beeswarm"] = path_summary
    print(f"        Saved: {path_summary}")
    
    # --- Plot 2: Bar Plot (Global Feature Importance) ---
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("SHAP Global Feature Importance (Top-20)", fontsize=14, pad=20)
    plt.tight_layout()
    path_bar = os.path.join(output_dir, "shap_feature_importance_bar.png")
    plt.savefig(path_bar, dpi=200, bbox_inches='tight')
    plt.close()
    saved_plots["importance_bar"] = path_bar
    print(f"        Saved: {path_bar}")
    
    # --- Plot 3: Waterfall Plot (Single Attack Prediction) ---
    # Find an Attack sample (label=1)
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
        print(f"        Saved: {path_wf_attack}")
    
    # --- Plot 4: Waterfall Plot (Single Benign Prediction) ---
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
        print(f"        Saved: {path_wf_benign}")
    
    # --- Plot 5: SHAP Importance Table (Top-20 features) ---
    print("  [5/5] Generating SHAP feature ranking table...")
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_SHAP_Value": mean_abs_shap
    }).sort_values("Mean_SHAP_Value", ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "shap_feature_importance.csv")
    feature_importance.to_csv(csv_path, index=False)
    saved_plots["importance_csv"] = csv_path
    
    # Print Top-20
    print(f"\n  {'─' * 50}")
    print(f"  Top-20 Features by Mean |SHAP Value|")
    print(f"  {'─' * 50}")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    print(f"  {'─' * 50}")
    for rank, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        print(f"  {rank:<6} {row['Feature']:<35} {row['Mean_SHAP_Value']:>12.6f}")
    
    print(f"\n  SHAP analysis completed! {len(saved_plots)} outputs saved to: {output_dir}")
    
    return saved_plots
