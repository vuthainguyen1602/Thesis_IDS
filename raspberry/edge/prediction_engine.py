#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time

from pyspark.ml import PipelineModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH


class PredictionEngine:

    LABEL_MAP = {0: "Benign", 1: "Attack"}

    def __init__(self, spark, model_path=None):
        self.spark = spark
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self.total_predictions = 0
        self.total_attacks = 0
        self.total_inference_time = 0.0

        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = PipelineModel.load(self.model_path)
            print(f"[OK] PySpark Model loaded from {self.model_path}")
            print(f"  Pipeline stages: {[type(s).__name__ for s in self.model.stages]}")
        else:
            raise FileNotFoundError(
                f"[ERR] Model not found at {self.model_path}\n"
                f"  Run scripts/save_model.py first to save the trained model."
            )

    def predict(self, spark_df):
        start = time.perf_counter()
        predictions = self.model.transform(spark_df)
        inference_time = (time.perf_counter() - start) * 1000

        count = predictions.count()
        attack_count = predictions.filter(predictions.prediction == 1.0).count()

        self.total_predictions += count
        self.total_attacks += attack_count
        self.total_inference_time += inference_time

        stats = {
            "batch_size": count,
            "attacks_found": attack_count,
            "inference_time_ms": round(inference_time, 3),
            "avg_time_ms": round(inference_time / count, 3) if count > 0 else 0,
        }

        return predictions, stats

    def predict_single(self, spark_df):
        start = time.perf_counter()
        predictions = self.model.transform(spark_df)
        row = predictions.select("prediction").first()
        inference_time = (time.perf_counter() - start) * 1000

        prediction = int(row["prediction"])
        self.total_predictions += 1
        self.total_inference_time += inference_time
        if prediction == 1:
            self.total_attacks += 1

        return {
            "prediction": prediction,
            "label": self.LABEL_MAP.get(prediction, "Unknown"),
            "inference_time_ms": round(inference_time, 3),
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
