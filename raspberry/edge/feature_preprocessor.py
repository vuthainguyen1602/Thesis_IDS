#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Preprocessor \u2014 Raspberry Pi Edge IDS (PySpark).

Converts raw JSON network-flow data into a Spark DataFrame suitable for
the PySpark PipelineModel.  The pipeline already contains VectorAssembler +
StandardScaler stages, so this module only handles data cleaning and
DataFrame construction.

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import json

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_PATH, SHAP_TOP_FEATURES


class FeaturePreprocessor:
    """
    Preprocesses raw network flow data for the PySpark prediction pipeline.
    - Creates Spark DataFrame from JSON message
    - Handles missing/infinity values
    - The VectorAssembler + StandardScaler are already in the PipelineModel
    """

    def __init__(self, spark, features_path=None):
        """
        Args:
            spark: Active SparkSession
            features_path: Path to feature_columns.json
        """
        self.spark = spark
        self.features_path = features_path or FEATURES_PATH
        self.feature_columns = SHAP_TOP_FEATURES

        self._load_feature_columns()
        self._build_schema()

    def _load_feature_columns(self):
        """Load the feature column list from JSON."""
        if os.path.exists(self.features_path):
            with open(self.features_path, "r") as f:
                self.feature_columns = json.load(f)
            print(f"[OK] Feature columns loaded: {len(self.feature_columns)} features")
        else:
            print(f"[WARN] Using default SHAP_TOP_FEATURES ({len(self.feature_columns)} features)")

    def _build_schema(self):
        """Build Spark StructType schema for the feature columns."""
        fields = []
        for col_name in self.feature_columns:
            fields.append(StructField(col_name, DoubleType(), True))
        self.schema = StructType(fields)

    def clean_value(self, value):
        """Convert a raw value to a clean float."""
        if value is None:
            return 0.0
        try:
            v = float(value)
            if v != v or v == float("inf") or v == float("-inf"):  # NaN or Inf
                return 0.0
            return v
        except (ValueError, TypeError):
            return 0.0

    def preprocess(self, raw_data: dict):
        """
        Convert a single raw JSON message to a single-row Spark DataFrame.

        Args:
            raw_data: dict with feature_name -> value

        Returns:
            Spark DataFrame with feature columns, ready for PipelineModel
        """
        row = []
        for col_name in self.feature_columns:
            row.append(self.clean_value(raw_data.get(col_name, 0.0)))

        df = self.spark.createDataFrame([tuple(row)], schema=self.schema)
        return df

    def preprocess_batch(self, raw_data_list: list):
        """
        Convert a batch of raw JSON messages to a Spark DataFrame.

        Args:
            raw_data_list: list of dicts

        Returns:
            Spark DataFrame with feature columns
        """
        rows = []
        for raw_data in raw_data_list:
            row = []
            for col_name in self.feature_columns:
                row.append(self.clean_value(raw_data.get(col_name, 0.0)))
            rows.append(tuple(row))

        df = self.spark.createDataFrame(rows, schema=self.schema)
        return df
