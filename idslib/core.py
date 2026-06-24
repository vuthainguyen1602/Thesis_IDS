#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import random
import logging
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

logger = logging.getLogger(__name__)

import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def _configure_java_home() -> None:
    def _java_home_valid(path: str) -> bool:
        return bool(path) and os.path.isfile(os.path.join(path, "bin", "java"))

    if _java_home_valid(os.environ.get("JAVA_HOME", "")):
        java_home = os.environ["JAVA_HOME"]
    elif sys.platform == "darwin":
        candidates = [
            "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
            "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home",
        ]
        java_home = next((p for p in candidates if _java_home_valid(p)), candidates[-1])
        os.environ["JAVA_HOME"] = java_home
    else:
        import shutil

        java_bin = shutil.which("java")
        if java_bin:
            java_home = os.path.dirname(os.path.dirname(os.path.realpath(java_bin)))
        else:
            java_home = ""
        if not _java_home_valid(java_home):
            candidates = [
                "/usr/lib/jvm/java-17-openjdk-arm64",
                "/usr/lib/jvm/java-11-openjdk-arm64",
                "/usr/lib/jvm/default-java",
                "/usr/lib/jvm/java-17-openjdk-amd64",
                "/usr/lib/jvm/java-11-openjdk-amd64",
            ]
            java_home = next((p for p in candidates if _java_home_valid(p)), "")
        if not _java_home_valid(java_home):
            raise RuntimeError(
                "JAVA_HOME not found. On Jetson run: sudo apt-get install -y default-jdk"
            )
        os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")


_configure_java_home()


def _allow_local_spark() -> bool:
    return os.environ.get("IDS_ALLOW_LOCAL_SPARK", "").strip().lower() in ("1", "true", "yes")


def require_distributed_spark(context: str = "this script") -> None:
    master = os.environ.get("SPARK_MASTER", "")
    if master.startswith("spark://"):
        return
    if _allow_local_spark():
        return
    raise RuntimeError(
        f"Distributed Spark required for {context}.\n"
        "  1. cp cluster/spark_cluster.env.example cluster/spark_cluster.env\n"
        "  2. Edit IPs, then: ./cluster/reproduce_cluster.sh fair|thesis|soict\n"
        "  Or: source cluster/load_cluster_env.sh (Mac/Jetson with cluster up)\n"
        f"  Current SPARK_MASTER={master!r}"
    )


_SPARK_MASTER = os.environ.get("SPARK_MASTER", "")
if _SPARK_MASTER.startswith("spark://") or _allow_local_spark():
    if _SPARK_MASTER and not _SPARK_MASTER.startswith("spark://"):
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--master {_SPARK_MASTER} pyspark-shell"
else:
    _SPARK_MASTER = ""

from pyspark.sql import SparkSession, functions as F
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

def _is_arm64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


# LightGBM (via SynapseML) was removed from the algorithm set: its native libs
# are x86_64-only and do not run on the Jetson Orin Nano Super (ARM64) edge target, so a
# LightGBM model could never be deployed in our pipeline. Keeping it would also
# make x86 vs. ARM result sets inconsistent. HAS_LIGHTGBM stays defined as False
# for backward compatibility with any guarded references.
HAS_LIGHTGBM = False

try:
    from xgboost.spark import SparkXGBClassifier
    HAS_XGBOOST = True
    print("[INFO] XGBoost backend available")
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not available (pip install xgboost pyarrow)")


_DEFAULT_DATA_DIR: str = os.environ.get(
    "IDS_DATA_DIR",
    os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "data")
)

ML01_DIR = "ml_01_baseline"
ML02_DIR = "ml_02_feature_selection_rf"
ML03_DIR = "ml_03_hyperparameter_tuning"
ML04_DIR = "ml_04_pca"
ML05_DIR = "ml_05_shap_explainability"
ML06_DIR = "ml_06_feature_selection_shap"
ML07_DIR = "ml_07_cross_method_comparison"
ML08_DIR = "ml_08_anomaly_gate"
ML09_DIR = "ml_09_multiclass_eval"


def ids_root() -> str:
    return os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__)))


def ml_results_dir(experiment_dir: str, *parts: str, mkdir: bool = True) -> str:
    """Build path under results/<experiment_dir>/ and optionally create parent dirs."""
    path = os.path.join(ids_root(), "results", experiment_dir, *parts)
    if mkdir:
        os.makedirs(os.path.dirname(path) if parts else path, exist_ok=True)
    return path


def shared_results_path(filename: str) -> str:
    path = os.path.join(ids_root(), "results", "shared", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def is_cluster_mode(master=None) -> bool:
    master = master or os.environ.get("SPARK_MASTER", "")
    return master.startswith("spark://")


def resolve_data_dir() -> str:
    """Return parquet directory visible to Spark executors on all cluster nodes."""
    if is_cluster_mode() and not _allow_local_spark():
        cluster_dir = os.environ.get("IDS_CLUSTER_DATA_DIR")
        if not cluster_dir:
            raise ValueError(
                "IDS_CLUSTER_DATA_DIR is required in distributed mode. "
                "Set it in cluster/spark_cluster.env and run cluster/sync_workspace.sh."
            )
        return cluster_dir
    return _DEFAULT_DATA_DIR


def create_spark_session(app_name: str = "IDS_Binary_Prediction") -> SparkSession:
    require_distributed_spark(app_name)
    if _allow_local_spark():
        # Mac-only scripts (ml_00, save_model): raw CSV / local paths — never use cluster.
        master = "local[*]"
        cluster = False
    else:
        master = os.environ.get("SPARK_MASTER", "")
        cluster = is_cluster_mode(master)

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "100s")
    )

    # No SynapseML/LightGBM JAR configuration — LightGBM is not used on the
    # ARM64 Jetson target.

    if cluster:
        driver_host = os.environ.get("SPARK_DRIVER_HOST")
        if driver_host:
            builder = (
                builder
                .config("spark.driver.host", driver_host)
                .config("spark.driver.bindAddress", "0.0.0.0")
            )
        builder = (
            builder
            .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", "3g"))
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "3g"))
            .config("spark.executor.cores", os.environ.get("SPARK_EXECUTOR_CORES", "4"))
            .config("spark.driver.maxResultSize", os.environ.get("SPARK_DRIVER_MAX_RESULT_SIZE", "3g"))
            .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "32"))
            .config("spark.memory.fraction", "0.75")
        )
        print(f"[INFO] Spark cluster mode | master={master}")
    else:
        print(f"[INFO] Spark local mode | master={master}")
        builder = (
            builder
            .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", "8g"))
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "8g"))
            .config("spark.memory.fraction", "0.8")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    ui = spark.sparkContext.uiWebUrl or "(cluster — see master UI)"
    print(f"[INFO] Spark {spark.version} | UI: {ui}")
    return spark



GLOBAL_SEED = int(os.environ.get("IDS_GLOBAL_SEED", "42"))

# Environment switches that change scientific results — logged at every run so
# a result file can always be traced back to the exact configuration.
_REPRO_ENV_KEYS = [
    "IDS_GLOBAL_SEED", "IDS_KEEP_PORT_FEATURES", "IDS_DEDUP_ON_FEATURES",
    "IDS_SPLIT_MODE", "IDS_ROBUST_DATA_DIR", "IDS_STAT_SPLITS",
    "IDS_CV_FOLDS", "IDS_CV_FRACTION", "IDS_EXP2_FULL", "SPARK_MASTER",
]


def log_run_config(app_name: str, extra: dict = None) -> None:
    """Print a reproducibility banner: versions, seed and result-affecting env.

    Call once at the start of every experiment so each output CSV/figure can be
    traced to the exact code configuration that produced it.
    """
    import platform
    print("\n" + "#" * 70)
    print(f"#  RUN CONFIG — {app_name}")
    print("#" * 70)
    print(f"  timestamp      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  python         : {platform.python_version()}")
    try:
        import pyspark
        print(f"  pyspark        : {pyspark.__version__}")
    except Exception:
        pass
    print(f"  GLOBAL_SEED    : {GLOBAL_SEED}")
    for k in _REPRO_ENV_KEYS:
        v = os.environ.get(k)
        if v is not None:
            print(f"  {k:<22}: {v}")
    if extra:
        for k, v in extra.items():
            print(f"  {k:<22}: {v}")
    print("#" * 70 + "\n")

