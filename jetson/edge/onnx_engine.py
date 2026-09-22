#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JVM-free classifier tier: ONNX Runtime.

Consumes the graph produced by ``scripts/export_onnx.py``, which folds the
StandardScaler and the RandomForest of the deployed Spark PipelineModel into a
single ONNX graph. No PySpark import anywhere in this path -- with the
autoencoder gate already running on joblib/numpy (``anomaly_scorer.py``), an
all-ONNX deployment takes the JVM off the board entirely.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ONNX_MODEL_PATH, ONNX_PROVIDERS, ONNX_INTRA_THREADS
from edge.inference_engine import InferenceEngine


class OnnxInferenceEngine(InferenceEngine):

    def __init__(self, model_path=None, providers=None):
        super().__init__()
        self.model_path = model_path or ONNX_MODEL_PATH
        self.session = None
        self.input_name = None
        self.label_name = None
        self.prob_name = None

        self._load_model(providers)

    def _load_model(self, providers):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "[ERR] EDGE_ENGINE='onnx' needs onnxruntime: pip install onnxruntime"
            ) from e

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[ERR] ONNX model not found at {self.model_path}\n"
                f"  Run scripts/export_onnx.py to convert the Spark PipelineModel first."
            )

        opts = ort.SessionOptions()
        if ONNX_INTRA_THREADS > 0:
            opts.intra_op_num_threads = ONNX_INTRA_THREADS

        requested = providers or ONNX_PROVIDERS
        available = ort.get_available_providers()
        chosen = [p for p in requested if p in available] or ["CPUExecutionProvider"]
        if chosen != list(requested):
            print(f"[WARN] Providers {requested} unavailable, using {chosen}")

        self.session = ort.InferenceSession(self.model_path, opts, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        outputs = [o.name for o in self.session.get_outputs()]
        self.label_name, self.prob_name = outputs[0], outputs[1]

        n_features = self.session.get_inputs()[0].shape[-1]
        print(f"[OK] ONNX model loaded from {self.model_path}")
        print(f"  Providers: {self.session.get_providers()} | features: {n_features}")

    def _infer(self, matrix):
        x = np.ascontiguousarray(matrix, dtype=np.float32)
        labels, probs = self.session.run(
            [self.label_name, self.prob_name], {self.input_name: x})

        preds = np.asarray(labels).astype(np.int64).reshape(-1)
        # TreeEnsembleClassifier emits a (n, c) score tensor when class labels
        # are int64, so no ZipMap unpacking is needed here.
        probs = np.asarray(probs, dtype=np.float64).reshape(len(preds), -1)
        return preds, probs
