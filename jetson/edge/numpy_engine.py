#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JVM-free classifier tier: pure NumPy RandomForest inference.

Consumes the artifact produced by ``scripts/export_numpy.py``. The artifact
folds the fitted Spark ``StandardScalerModel`` parameters together with the
saved RandomForest node table, so runtime inference only needs a raw feature
matrix in the same column order as ``feature_columns.json``.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUMPY_MODEL_PATH
from edge.inference_engine import InferenceEngine


class NumpyInferenceEngine(InferenceEngine):

    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path or NUMPY_MODEL_PATH
        self.meta = {}
        self.offset = None
        self.scale = None
        self.tree_offsets = None
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.is_leaf = None
        self.leaf_scores = None
        self.n_classes = 0

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[ERR] NumPy model not found at {self.model_path}\n"
                f"  Run scripts/export_numpy.py to export the Spark PipelineModel first."
            )

        data = np.load(self.model_path, allow_pickle=False)
        self.meta = json.loads(str(data["metadata_json"]))
        self.offset = data["offset"].astype(np.float64, copy=False)
        self.scale = data["scale"].astype(np.float64, copy=False)
        self.tree_offsets = data["tree_offsets"].astype(np.int64, copy=False)
        self.left = data["left"].astype(np.int64, copy=False)
        self.right = data["right"].astype(np.int64, copy=False)
        self.feature = data["feature"].astype(np.int64, copy=False)
        self.threshold = data["threshold"].astype(np.float64, copy=False)
        self.is_leaf = data["is_leaf"].astype(bool, copy=False)
        self.leaf_scores = data["leaf_scores"].astype(np.float64, copy=False)
        self.n_classes = int(self.leaf_scores.shape[1])

        print(f"[OK] NumPy model loaded from {self.model_path}")
        print(
            f"  Trees: {len(self.tree_offsets) - 1} | "
            f"features: {len(self.offset)} | classes: {self.n_classes}"
        )

    def _infer(self, matrix):
        x = np.ascontiguousarray(matrix, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.offset):
            raise ValueError(
                f"[ERR] Expected matrix shape (n, {len(self.offset)}), got {x.shape}"
            )

        scaled = (x - self.offset) * self.scale
        scores = np.zeros((scaled.shape[0], self.n_classes), dtype=np.float64)

        for t in range(len(self.tree_offsets) - 1):
            base = int(self.tree_offsets[t])
            end = int(self.tree_offsets[t + 1])
            node = np.full(scaled.shape[0], base, dtype=np.int64)

            active = np.ones(scaled.shape[0], dtype=bool)
            while np.any(active):
                rows = np.flatnonzero(active)
                cur = node[rows]
                leaf = self.is_leaf[cur]
                if np.any(leaf):
                    leaf_rows = rows[leaf]
                    scores[leaf_rows] += self.leaf_scores[node[leaf_rows]]
                    active[leaf_rows] = False

                branch_rows = rows[~leaf]
                if branch_rows.size:
                    cur_branch = node[branch_rows]
                    feat = self.feature[cur_branch]
                    go_left = scaled[branch_rows, feat] <= self.threshold[cur_branch]
                    node[branch_rows] = np.where(
                        go_left, self.left[cur_branch], self.right[cur_branch])

                if np.any((node < base) | (node >= end)):
                    raise RuntimeError("[ERR] Tree traversal left the exported node range")

        preds = np.argmax(scores, axis=1).astype(np.int64)
        return preds, scores
