#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swappable classifier tier.

The SOICT deployment serves the exported Spark ``PipelineModel`` on the board,
which drags the JVM onto an 8 GB ARM device; the paper measures that cost and
leaves an ONNX/TensorRT serving path as future work. This module is that path:
one contract, multiple backends, selected by ``EDGE_ENGINE=spark|numpy|onnx``.

The Spark backend is kept so the published benchmark numbers stay reproducible
from the same tree -- do not delete it.

Contract: engines take an already-assembled float matrix (see
``feature_matrix.FeatureMatrixBuilder``) and return per-row verdicts, so no
engine-specific frame type leaks into the pipeline roles.
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EDGE_ENGINE


@dataclass(frozen=True)
class BatchPrediction:
    predictions: np.ndarray   # int labels, shape (n,)
    confidences: np.ndarray   # probability of the predicted class, shape (n,)
    stats: dict               # batch_size, attacks_found, inference_time_ms, avg_time_ms


class InferenceEngine:
    """Base class: counters and the stats shape shared by every backend."""

    LABEL_MAP = {0: "Benign", 1: "Attack"}

    def __init__(self):
        self.total_predictions = 0
        self.total_attacks = 0
        self.total_inference_time = 0.0

    # --- to implement in a backend -------------------------------------
    def _infer(self, matrix):
        """ndarray (n, f) -> (predictions int64 (n,), probabilities float (n, c))."""
        raise NotImplementedError

    # --- shared ---------------------------------------------------------
    def predict_batch(self, matrix):
        start = time.perf_counter()
        preds, probs = self._infer(matrix)
        inference_time = (time.perf_counter() - start) * 1000

        count = int(preds.shape[0])
        attacks = int(np.count_nonzero(preds == 1))
        confidences = (probs[np.arange(count), preds].astype(float)
                       if count and probs is not None else np.zeros(count))

        self.total_predictions += count
        self.total_attacks += attacks
        self.total_inference_time += inference_time

        return BatchPrediction(
            predictions=preds,
            confidences=confidences,
            stats={
                "batch_size": count,
                "attacks_found": attacks,
                "inference_time_ms": round(inference_time, 3),
                "avg_time_ms": round(inference_time / count, 3) if count > 0 else 0,
            },
        )

    def predict_single(self, matrix):
        r = self.predict_batch(matrix[:1])
        prediction = int(r.predictions[0])
        return {
            "prediction": prediction,
            "label": self.LABEL_MAP.get(prediction, "Unknown"),
            "confidence": float(r.confidences[0]),
            "inference_time_ms": r.stats["inference_time_ms"],
            "is_attack": prediction == 1,
        }

    def get_stats(self) -> dict:
        avg_time = (self.total_inference_time / self.total_predictions
                    if self.total_predictions > 0 else 0)
        return {
            "total_predictions": self.total_predictions,
            "total_attacks": self.total_attacks,
            "attack_rate": (self.total_attacks / self.total_predictions
                            if self.total_predictions > 0 else 0),
            "avg_inference_time_ms": round(avg_time, 3),
            "total_inference_time_ms": round(self.total_inference_time, 3),
        }

    def close(self):
        pass


def create_inference_engine(spark=None, engine=None):
    """Imports lazily: JVM-free engines must not pull PySpark into the process."""
    name = (engine or EDGE_ENGINE).strip().lower()

    if name == "numpy":
        from edge.numpy_engine import NumpyInferenceEngine
        return NumpyInferenceEngine()

    if name == "onnx":
        from edge.onnx_engine import OnnxInferenceEngine
        return OnnxInferenceEngine()

    if name != "spark":
        print(f"[WARN] Unknown EDGE_ENGINE='{name}', falling back to 'spark'")

    from edge.spark_engine import SparkInferenceEngine
    if spark is None:
        raise ValueError("[ERR] EDGE_ENGINE='spark' requires an active SparkSession")
    return SparkInferenceEngine(spark)
