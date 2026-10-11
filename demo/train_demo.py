#!/usr/bin/env python
"""Phase-A demo: train the deployed model on the Spark Standalone cluster.

Uses the thesis library as the ML scripts do: leakage-aware train/test split
(port features excluded, feature-level duplicates already removed by ml_00),
class weights, and the Random Forest of the experiments (200 trees, depth 15)
on the SHAP Top-30 features from ml_06. Trains on the full training split and
evaluates on the full test split, so the printed metrics are comparable with
the thesis. Run through cluster/run_ml_remote.sh so it gets the cluster env:

    ./cluster/run_ml_remote.sh demo/train_demo.py
"""
import json
import os
import sys
import time
import urllib.request

# demo/ is not the repository root; make idslib importable as for the ml_*.py scripts
sys.path.insert(0, os.environ.get("IDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler

from idslib.core import create_spark_session
from idslib.data import load_and_prepare_data
from idslib.modeling import add_class_weights, get_classifiers, train_and_evaluate

ROOT = os.environ.get("IDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHAP_CSV = os.path.join(ROOT, "results", "ml_06_feature_selection_shap", "shap_feature_importance.csv")


def main():
    t_all = time.time()
    spark = create_spark_session("IDS_Demo_Distributed_Training")
    sc = spark.sparkContext
    execs = []
    for _ in range(30):  # executors register a few seconds after the session starts
        url = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/executors"
        execs = [e for e in json.load(urllib.request.urlopen(url)) if e["id"] != "driver"]
        if len({e["hostPort"].split(":")[0] for e in execs}) >= 2:
            break
        time.sleep(1)
    print("=" * 60)
    print("  SPARK CLUSTER")
    print("=" * 60)
    print(f"  Master:   {sc.master}")
    print(f"  App:      {sc.appName} ({sc.applicationId})")
    print(f"  Driver:   {sc.getConf().get('spark.driver.host')}  (UI {sc.uiWebUrl})")
    for e in sorted(execs, key=lambda e: e["hostPort"]):
        print(f"  Executor {e['id']}: {e['hostPort'].split(':')[0]:15} {e['totalCores']} cores, "
              f"{e['maxMemory'] / 2**30:.1f} GiB storage memory")

    _, train_df, test_df, feature_cols = load_and_prepare_data(spark)
    shap = pd.read_csv(SHAP_CSV)
    top30 = [f for f in shap["feature"] if f in feature_cols][:30]
    print("=" * 60)
    print("  FEATURE SELECTION: SHAP TOP-30 (from ml_06)")
    print("=" * 60)
    print(f"  {len(feature_cols)} leak-free features -> {len(top30)} selected")
    for i in range(0, len(top30), 3):
        print("  " + "  ".join(f"{f:30}" for f in top30[i:i + 3]))

    train_df = add_class_weights(train_df)
    rf = get_classifiers(features_col="features_scaled", num_features=len(top30))["Random Forest"]
    print("=" * 60)
    print("  TRAINING: Random Forest (200 trees, max depth 15) on the cluster")
    print("=" * 60)
    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=top30, outputCol="features_raw", handleInvalid="keep"),
        StandardScaler(inputCol="features_raw", outputCol="features_scaled",
                       withStd=True, withMean=True),
        rf,
    ])
    _, _, metrics = train_and_evaluate(pipeline, train_df, test_df,
                                       title="Random Forest - SHAP Top-30 (full test split)")
    print(f"\n[OK] Done in {time.time() - t_all:.0f} s")
    spark.stop()


if __name__ == "__main__":
    main()
