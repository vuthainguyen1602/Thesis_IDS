#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import signal

from kafka import KafkaConsumer
from pyspark.sql import SparkSession

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_GROUP_ID, ALERT_COOLDOWN,
    ANOMALY_ENABLED, ANOMALY_MODEL_PATH, ANOMALY_SCALER_PATH, ANOMALY_THRESHOLD_PATH, FEATURES_PATH,
    EDGE_BATCH_SIZE, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
)
from edge.feature_preprocessor import FeaturePreprocessor
from edge.anomaly_scorer import AnomalyScorer
from edge.prediction_engine import PredictionEngine
from edge.performance_monitor import PerformanceMonitor

try:
    from storage.postgres_storage import PostgresStorage
except ImportError:
    PostgresStorage = None

try:
    from storage.influxdb_storage import InfluxDBStorage
except ImportError:
    InfluxDBStorage = None

try:
    from alerting.alert_system import AlertSystem
except ImportError:
    AlertSystem = None


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("IDS_Edge_RaspberryPi")
        .master(SPARK_MASTER)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"[OK] Spark Session created (version: {spark.version})")
    return spark


class IDSEdgePipeline:

    BATCH_SIZE = EDGE_BATCH_SIZE

    def __init__(self):
        self.running = True
        self.last_alert_time = 0

        print("\n" + "=" * 60)
        print("  INITIALIZING IDS EDGE PIPELINE (PySpark)")
        print("=" * 60)

        self.spark = create_spark_session()

        self.preprocessor = FeaturePreprocessor(self.spark)
        self.anomaly = None
        if ANOMALY_ENABLED:
            try:
                self.anomaly = AnomalyScorer(
                    features_path=FEATURES_PATH,
                    model_path=ANOMALY_MODEL_PATH,
                    scaler_path=ANOMALY_SCALER_PATH,
                    threshold_path=ANOMALY_THRESHOLD_PATH,
                )
            except Exception as e:
                print(f"[WARN] AnomalyScorer disabled: {e}")
        self.engine = PredictionEngine(self.spark)

        self.postgres = None
        self.influxdb = None
        self.alerting = None

        if PostgresStorage:
            try:
                self.postgres = PostgresStorage()
                self.postgres.init_tables()
            except Exception as e:
                print(f"[WARN] PostgreSQL disabled: {e}")

        if InfluxDBStorage:
            try:
                self.influxdb = InfluxDBStorage()
            except Exception as e:
                print(f"[WARN] InfluxDB disabled: {e}")

        if AlertSystem:
            try:
                self.alerting = AlertSystem()
            except Exception as e:
                print(f"[WARN] Alert System disabled: {e}")

        self.monitor = PerformanceMonitor(influxdb_storage=self.influxdb)
        self.monitor.start()

        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        print(f"[OK] Kafka Consumer subscribed to '{KAFKA_TOPIC}'")

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n" + "=" * 60)
        print("  IDS EDGE PIPELINE READY (PySpark) - Waiting for messages...")
        print("=" * 60 + "\n")

    def _signal_handler(self, signum, frame):
        print("\n[WARN] Shutdown signal received...")
        self.running = False

    def process_batch(self, messages: list):
        timestamp = time.time()

        anomaly_time_ms = 0.0
        messages_to_classify = messages
        skipped_benign = 0
        anomaly_scores = None
        anomaly_flags = None
        anomaly_threshold = None
        suspicious_items = []
        if self.anomaly:
            r = self.anomaly.score_batch(messages)
            anomaly_time_ms = r.inference_time_ms
            anomaly_scores = r.scores
            anomaly_flags = r.is_anomaly
            anomaly_threshold = float(r.threshold)
            flags = r.is_anomaly.tolist()
            suspicious_items = [
                (m, i, float(anomaly_scores[i]))
                for i, (m, f) in enumerate(zip(messages, flags))
                if f
            ]
            messages_to_classify = [t[0] for t in suspicious_items]
            skipped_benign = len(messages) - len(messages_to_classify)

        predictions_df = None
        stats = {
            "batch_size": len(messages),
            "attacks_found": 0,
            "inference_time_ms": 0.0,
            "avg_time_ms": 0.0,
            "anomaly_time_ms": round(anomaly_time_ms, 3),
            "classified_size": len(messages_to_classify),
            "skipped_benign": skipped_benign,
        }

        if messages_to_classify:
            spark_df = self.preprocessor.preprocess_batch(messages_to_classify)
            predictions_df, clf_stats = self.engine.predict(spark_df)
            stats["attacks_found"] = clf_stats["attacks_found"]
            stats["inference_time_ms"] = clf_stats["inference_time_ms"]
            stats["avg_time_ms"] = clf_stats["avg_time_ms"]

        if (not self.anomaly) and messages_to_classify:
            suspicious_items = [(m, i, None) for i, m in enumerate(messages_to_classify)]

        avg_time = float(stats["avg_time_ms"])
        per_item_time = avg_time
        if stats["batch_size"] > 0 and anomaly_time_ms > 0:
            per_item_time = (stats["inference_time_ms"] + anomaly_time_ms) / stats["batch_size"]

        for _ in range(stats["batch_size"]):
            self.monitor.record_prediction(inference_time_ms=per_item_time, is_attack=False)

        if stats["attacks_found"] > 0:
            for _ in range(stats["attacks_found"]):
                self.monitor.record_prediction(inference_time_ms=0, is_attack=True)

            now = time.time()
            if now - self.last_alert_time > ALERT_COOLDOWN:
                self.last_alert_time = now
                self._send_alert(stats)

        if self.postgres:
            try:
                if predictions_df is not None:
                    results = predictions_df.select("prediction", "probability").collect()
                    for (msg, idx, score), row in zip(suspicious_items, results):
                        pred = int(row["prediction"])
                        prob = row["probability"]
                        confidence = float(prob[int(pred)]) if prob else 0.0
                        raw_features = None
                        if self.anomaly is not None and anomaly_scores is not None and anomaly_flags is not None:
                            raw_features = {
                                "route": "spark_classifier",
                                "anomaly_score": float(score),
                                "anomaly_flag": bool(anomaly_flags[idx]),
                                "anomaly_threshold": float(anomaly_threshold),
                            }
                        self.postgres.store_prediction(
                            timestamp=timestamp + (idx * 1e-6),
                            prediction=pred,
                            confidence=confidence,
                            label="Attack" if pred == 1 else "Benign",
                            inference_time_ms=avg_time,
                            raw_features=raw_features,
                        )

                if self.anomaly is not None and anomaly_scores is not None and anomaly_flags is not None:
                    for i, msg in enumerate(messages):
                        if bool(anomaly_flags[i]):
                            continue
                        s = float(anomaly_scores[i])
                        thr = float(anomaly_threshold)
                        conf = max(0.0, min(1.0, 1.0 - (s / thr))) if thr > 0 else 0.0
                        raw_features = {
                            "route": "anomaly_gate_only",
                            "anomaly_score": s,
                            "anomaly_flag": False,
                            "anomaly_threshold": thr,
                        }
                        self.postgres.store_prediction(
                            timestamp=timestamp + (i * 1e-6),
                            prediction=0,
                            confidence=conf,
                            label="Benign (Gate)",
                            inference_time_ms=avg_time,
                            raw_features=raw_features,
                        )
            except Exception as e:
                print(f"  [WARN] DB store error: {e}")

        return stats

    def _send_alert(self, stats):
        engine_stats = self.engine.get_stats()
        message = (
            f"[ALERT] ATTACK DETECTED\n"
            f"Batch attacks: {stats['attacks_found']}/{stats['batch_size']}\n"
            f"Total attacks: {engine_stats['total_attacks']}\n"
            f"Attack rate: {engine_stats['attack_rate']:.2%}\n"
            f"Avg latency: {engine_stats['avg_inference_time_ms']:.1f}ms"
        )

        attack_confidence = engine_stats.get("attack_rate", 0.0)

        if self.alerting:
            try:
                self.alerting.send_all(message)
            except Exception as e:
                print(f"  [WARN] Alert error: {e}")

        if self.postgres:
            try:
                self.postgres.store_alert(
                    alert_type="ATTACK_DETECTED",
                    message=message,
                    confidence=float(stats["attacks_found"]) / stats["batch_size"],
                )
            except Exception:
                pass

    def run(self):
        total_processed = 0
        batch_buffer = []

        try:
            for message in self.consumer:
                if not self.running:
                    break

                batch_buffer.append(message.value)

                if len(batch_buffer) >= self.BATCH_SIZE:
                    stats = self.process_batch(batch_buffer)
                    total_processed += stats["batch_size"]
                    batch_buffer = []

                    if total_processed % 100 == 0:
                        engine_stats = self.engine.get_stats()
                        print(
                            f"  [{total_processed:,}] "
                            f"Batch: {stats['inference_time_ms']:.0f}ms | "
                            f"Attacks: {engine_stats['total_attacks']}"
                            f"/{engine_stats['total_predictions']} | "
                            f"Avg: {engine_stats['avg_inference_time_ms']:.1f}ms"
                        )

            if batch_buffer:
                self.process_batch(batch_buffer)
                total_processed += len(batch_buffer)

        except Exception as e:
            print(f"\n[ERR] Pipeline error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n" + "=" * 60)
        print("  SHUTTING DOWN IDS EDGE PIPELINE")
        print("=" * 60)

        self.monitor.stop()
        self.consumer.close()

        stats = self.engine.get_stats()
        print(f"\n  Final Statistics:")
        print(f"    Total predictions:  {stats['total_predictions']:,}")
        print(f"    Total attacks:      {stats['total_attacks']:,}")
        print(f"    Attack rate:        {stats['attack_rate']:.2%}")
        print(f"    Avg latency:        {stats['avg_inference_time_ms']:.3f} ms")

        if self.postgres:
            self.postgres.close()
        if self.influxdb:
            self.influxdb.close()

        self.spark.stop()
        print("\n[OK] Pipeline shutdown complete.")


if __name__ == "__main__":
    pipeline = IDSEdgePipeline()
    pipeline.run()
