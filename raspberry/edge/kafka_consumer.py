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
)
from edge.feature_preprocessor import FeaturePreprocessor
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
        .master("local[2]")
        .config("spark.executor.memory", "1g")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"[OK] Spark Session created (version: {spark.version})")
    return spark


class IDSEdgePipeline:

    BATCH_SIZE = 10

    def __init__(self):
        self.running = True
        self.last_alert_time = 0

        print("\n" + "=" * 60)
        print("  INITIALIZING IDS EDGE PIPELINE (PySpark)")
        print("=" * 60)

        self.spark = create_spark_session()

        self.preprocessor = FeaturePreprocessor(self.spark)
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

        spark_df = self.preprocessor.preprocess_batch(messages)

        predictions_df, stats = self.engine.predict(spark_df)

        avg_time = stats["avg_time_ms"]
        for _ in range(stats["batch_size"]):
            self.monitor.record_prediction(
                inference_time_ms=avg_time,
                is_attack=False,
            )

        if stats["attacks_found"] > 0:
            for _ in range(stats["attacks_found"]):
                self.monitor.record_prediction(inference_time_ms=0, is_attack=True)

            now = time.time()
            if now - self.last_alert_time > ALERT_COOLDOWN:
                self.last_alert_time = now
                self._send_alert(stats)

        if self.postgres:
            try:
                results = predictions_df.select("prediction", "probability").collect()
                for row in results:
                    pred = int(row["prediction"])
                    prob = row["probability"]
                    confidence = float(prob[int(pred)]) if prob else 0.0
                    self.postgres.store_prediction(
                        timestamp=timestamp,
                        prediction=pred,
                        confidence=confidence,
                        label="Attack" if pred == 1 else "Benign",
                        inference_time_ms=avg_time,
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
