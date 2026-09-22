#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Engine-agnostic feature assembly: Kafka message dicts -> float32 matrix.

Deliberately free of any PySpark import so the classifier tier can run without
a JVM on the board. ``FeaturePreprocessor`` (the Spark path) keeps its own
Spark-side assembly; both must agree on column order and cleaning, so the
column loading and the NaN/Inf policy live here as the single source of truth.
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_PATH, SHAP_TOP_FEATURES


def load_feature_columns(features_path=None, quiet=False):
    path = features_path or FEATURES_PATH
    if os.path.exists(path):
        with open(path, "r") as f:
            columns = json.load(f)
        if not quiet:
            print(f"[OK] Feature columns loaded: {len(columns)} features")
        return columns
    if not quiet:
        print(f"[WARN] Using default SHAP_TOP_FEATURES ({len(SHAP_TOP_FEATURES)} features)")
    return list(SHAP_TOP_FEATURES)


def clean_value(value):
    """Same policy as FeaturePreprocessor.clean_value: missing/NaN/Inf -> 0.0."""
    if value is None:
        return 0.0
    try:
        v = float(value)
        if v != v or v == float("inf") or v == float("-inf"):
            return 0.0
        return v
    except (ValueError, TypeError):
        return 0.0


class FeatureMatrixBuilder:

    def __init__(self, features_path=None, dtype=np.float32):
        self.feature_columns = load_feature_columns(features_path)
        self.dtype = dtype

    @property
    def n_features(self):
        return len(self.feature_columns)

    def build(self, raw_data_list):
        """list[dict] -> ndarray of shape (n_rows, n_features)."""
        cols = self.feature_columns
        out = np.empty((len(raw_data_list), len(cols)), dtype=self.dtype)
        for i, raw in enumerate(raw_data_list):
            for j, name in enumerate(cols):
                out[i, j] = clean_value(raw.get(name, 0.0))
        return out

    def build_single(self, raw_data):
        return self.build([raw_data])
