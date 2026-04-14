#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
from dataclasses import dataclass

import joblib
import numpy as np


@dataclass(frozen=True)
class AnomalyBatchResult:
    scores: np.ndarray
    is_anomaly: np.ndarray
    threshold: float
    inference_time_ms: float


class AnomalyScorer:
    def __init__(self, *, features_path: str, model_path: str, scaler_path: str, threshold_path: str):
        self.features_path = features_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold_path = threshold_path

        self.feature_columns: list[str] = []
        self.model = None
        self.scaler = None
        self.threshold = None

        self._load()

    def _load(self):
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"[ERR] Feature columns not found: {self.features_path}")
        with open(self.features_path, "r") as f:
            self.feature_columns = json.load(f)

        for p in (self.model_path, self.scaler_path, self.threshold_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[ERR] Anomaly artifact not found: {p}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        with open(self.threshold_path, "r") as f:
            payload = json.load(f)
        self.threshold = float(payload["threshold"])

        print(
            "[OK] AnomalyScorer loaded\n"
            f"  Features:   {len(self.feature_columns)}\n"
            f"  Threshold:  {self.threshold:.6f}\n"
            f"  Model:      {self.model_path}\n"
            f"  Scaler:     {self.scaler_path}"
        )

    @staticmethod
    def _clean_value(value) -> float:
        if value is None:
            return 0.0
        try:
            v = float(value)
            if v != v or v == float("inf") or v == float("-inf"):
                return 0.0
            return v
        except (ValueError, TypeError):
            return 0.0

    def _to_matrix(self, raw_data_list: list[dict]) -> np.ndarray:
        n = len(raw_data_list)
        d = len(self.feature_columns)
        x = np.zeros((n, d), dtype=np.float32)
        for i, raw in enumerate(raw_data_list):
            row = []
            for col in self.feature_columns:
                row.append(self._clean_value(raw.get(col, 0.0)))
            x[i, :] = np.asarray(row, dtype=np.float32)
        return x

    def score_batch(self, raw_data_list: list[dict]) -> AnomalyBatchResult:
        if not raw_data_list:
            return AnomalyBatchResult(
                scores=np.asarray([], dtype=np.float32),
                is_anomaly=np.asarray([], dtype=bool),
                threshold=float(self.threshold),
                inference_time_ms=0.0,
            )

        start = time.perf_counter()
        x = self._to_matrix(raw_data_list)
        x_scaled = self.scaler.transform(x)
        x_hat = self.model.predict(x_scaled)
        if x_hat.ndim == 1:
            x_hat = x_hat.reshape(-1, x_scaled.shape[1])
        mse = np.mean((x_scaled - x_hat) ** 2, axis=1)
        is_anomaly = mse >= float(self.threshold)
        ms = (time.perf_counter() - start) * 1000.0

        return AnomalyBatchResult(
            scores=mse.astype(np.float32),
            is_anomaly=is_anomaly.astype(bool),
            threshold=float(self.threshold),
            inference_time_ms=float(ms),
        )

