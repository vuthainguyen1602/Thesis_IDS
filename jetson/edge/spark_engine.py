#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spark backend behind the common engine contract.

Wraps the original ``PredictionEngine`` (left untouched, so the SOICT benchmark
path keeps running exactly as measured) and adapts its DataFrame in / DataFrame
out shape to the matrix in / verdicts out contract of ``InferenceEngine``.
"""

import os
import sys

import numpy as np
from pyspark.sql.types import StructType, StructField, DoubleType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.feature_matrix import load_feature_columns
from edge.inference_engine import InferenceEngine
from edge.prediction_engine import PredictionEngine


class SparkInferenceEngine(InferenceEngine):

    def __init__(self, spark, model_path=None, features_path=None):
        super().__init__()
        self.spark = spark
        self.engine = PredictionEngine(spark, model_path=model_path)
        self.feature_columns = load_feature_columns(features_path, quiet=True)
        self.schema = StructType(
            [StructField(c, DoubleType(), True) for c in self.feature_columns])

    def _infer(self, matrix):
        rows = [tuple(float(v) for v in row) for row in matrix]
        df = self.spark.createDataFrame(rows, schema=self.schema)
        predictions_df, _ = self.engine.predict(df)

        collected = predictions_df.select("prediction", "probability").collect()
        preds = np.array([int(r["prediction"]) for r in collected], dtype=np.int64)
        probs = np.array([list(r["probability"]) for r in collected], dtype=np.float64)
        return preds, probs
