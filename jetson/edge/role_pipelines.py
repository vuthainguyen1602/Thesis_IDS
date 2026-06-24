#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time

from kafka import KafkaConsumer
from pyspark.sql import SparkSession

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    KAFKA_SUSPICIOUS_TOPIC,
    KAFKA_GROUP_ID,
    KAFKA_CLASSIFIER_GROUP_ID,
    ANOMALY_ENABLED,
    ANOMALY_MODEL_PATH,
    ANOMALY_SCALER_PATH,
    ANOMALY_THRESHOLD_PATH,
    FEATURES_PATH,
    EDGE_BATCH_SIZE,
    EDGE_NODE_ID,
    EDGE_NODE_ROLE,
    SPARK_MASTER,
    SPARK_DRIVER_HOST,
    SPARK_EXECUTOR_MEMORY,
    SPARK_DRIVER_MEMORY,
    SPARK_SHUFFLE_PARTITIONS,
    SPARK_APP_NAME,
)
from edge.pipeline_base import PipelineBase
from edge.feature_preprocessor import FeaturePreprocessor
from edge.anomaly_scorer import AnomalyScorer
from edge.prediction_engine import PredictionEngine
from edge.kafka_forwarder import SuspiciousFlowForwarder


def create_spark_session():
    if not SPARK_MASTER.startswith("spark://"):
        raise RuntimeError(
            "SPARK_MASTER must be spark://<MAC_IP>:7077 in jetson/.env "
            "(copy from cluster/spark_cluster.env.example)."
        )
    builder = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
    )
    if SPARK_MASTER.startswith("spark://") and SPARK_DRIVER_HOST:
        builder = (
            builder
            .config("spark.driver.host", SPARK_DRIVER_HOST)
            .config("spark.driver.bindAddress", "0.0.0.0")
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print(f"[OK] Spark Session created (version: {spark.version}, master: {SPARK_MASTER})")
    return spark


class FullPipeline(PipelineBase):
    """Complete IDS pipeline. Use the same KAFKA_GROUP_ID on multiple nodes for horizontal scaling."""

    BATCH_SIZE = EDGE_BATCH_SIZE

    def __init__(self):
        print("\n" + "=" * 60)
        print(f"  IDS FULL PIPELINE ({EDGE_NODE_ID})")
        print("=" * 60)
        super().__init__()

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

        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        print(f"[OK] Kafka Consumer subscribed to '{KAFKA_TOPIC}' (group: {KAFKA_GROUP_ID})")
        print("\n" + "=" * 60)
        print("  PIPELINE READY - Waiting for messages...")
        print("=" * 60 + "\n")

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

            engine_stats = self.engine.get_stats()
            self._send_alert(
                (
                    f"[ALERT] ATTACK DETECTED ({EDGE_NODE_ID})\n"
                    f"Batch attacks: {stats['attacks_found']}/{stats['batch_size']}\n"
                    f"Total attacks: {engine_stats['total_attacks']}\n"
                    f"Attack rate: {engine_stats['attack_rate']:.2%}\n"
                    f"Avg latency: {engine_stats['avg_inference_time_ms']:.1f}ms"
                ),
                float(stats["attacks_found"]) / stats["batch_size"],
            )

        if self.postgres:
            try:
                if predictions_df is not None:
                    results = predictions_df.select("prediction", "probability").collect()
                    for (msg, idx, score), row in zip(suspicious_items, results):
                        pred = int(row["prediction"])
                        prob = row["probability"]
                        confidence = float(prob[int(pred)]) if prob else 0.0
                        raw_features = {"route": "spark_classifier"}
                        if self.anomaly is not None and anomaly_scores is not None and anomaly_flags is not None:
                            raw_features.update({
                                "anomaly_score": float(score),
                                "anomaly_flag": bool(anomaly_flags[idx]),
                                "anomaly_threshold": float(anomaly_threshold),
                            })
                        self._store_prediction(
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
                        self._store_prediction(
                            timestamp=timestamp + (i * 1e-6),
                            prediction=0,
                            confidence=conf,
                            label="Benign (Gate)",
                            inference_time_ms=avg_time,
                            raw_features={
                                "route": "anomaly_gate_only",
                                "anomaly_score": s,
                                "anomaly_flag": False,
                                "anomaly_threshold": thr,
                            },
                        )
            except Exception as e:
                print(f"  [WARN] DB store error: {e}")

        return stats

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
                            f"  [{EDGE_NODE_ID} {total_processed:,}] "
                            f"Batch: {stats['inference_time_ms']:.0f}ms | "
                            f"Attacks: {engine_stats['total_attacks']}"
                            f"/{engine_stats['total_predictions']} | "
                            f"Avg: {engine_stats['avg_inference_time_ms']:.1f}ms"
                        )
            if batch_buffer:
                self.process_batch(batch_buffer)
        except Exception as e:
            print(f"\n[ERR] Pipeline error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        if self.engine:
            stats = self.engine.get_stats()
            print(f"\n  Final Statistics ({EDGE_NODE_ID}):")
            print(f"    Total predictions:  {stats['total_predictions']:,}")
            print(f"    Total attacks:      {stats['total_attacks']:,}")
            print(f"    Attack rate:        {stats['attack_rate']:.2%}")
            print(f"    Avg latency:        {stats['avg_inference_time_ms']:.3f} ms")
        self._shutdown_common()


class AnomalyGatePipeline(PipelineBase):
    """Jetson #1 role: score all flows, forward suspicious ones to the classifier topic."""

    BATCH_SIZE = EDGE_BATCH_SIZE

    def __init__(self):
        print("\n" + "=" * 60)
        print(f"  IDS ANOMALY GATE ({EDGE_NODE_ID})")
        print("=" * 60)
        super().__init__()

        self.anomaly = AnomalyScorer(
            features_path=FEATURES_PATH,
            model_path=ANOMALY_MODEL_PATH,
            scaler_path=ANOMALY_SCALER_PATH,
            threshold_path=ANOMALY_THRESHOLD_PATH,
        )
        self.forwarder = SuspiciousFlowForwarder()
        self.total_scored = 0
        self.total_forwarded = 0

        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        print(f"[OK] Kafka Consumer subscribed to '{KAFKA_TOPIC}' (group: {KAFKA_GROUP_ID})")
        print(f"[OK] Suspicious flows -> '{KAFKA_SUSPICIOUS_TOPIC}'")
        print("\n" + "=" * 60)
        print("  ANOMALY GATE READY - Waiting for messages...")
        print("=" * 60 + "\n")

    def process_batch(self, messages: list):
        timestamp = time.time()
        r = self.anomaly.score_batch(messages)
        per_item_ms = r.inference_time_ms / len(messages) if messages else 0.0

        for _ in messages:
            self.monitor.record_prediction(inference_time_ms=per_item_ms, is_attack=False)

        forwarded = 0
        for i, msg in enumerate(messages):
            score = float(r.scores[i])
            is_anomaly = bool(r.is_anomaly[i])
            if is_anomaly:
                self.forwarder.forward(msg, anomaly_score=score, source_node=EDGE_NODE_ID)
                forwarded += 1
            elif self.postgres:
                thr = float(r.threshold)
                conf = max(0.0, min(1.0, 1.0 - (score / thr))) if thr > 0 else 0.0
                self._store_prediction(
                    timestamp=timestamp + (i * 1e-6),
                    prediction=0,
                    confidence=conf,
                    label="Benign (Gate)",
                    inference_time_ms=per_item_ms,
                    raw_features={
                        "route": "anomaly_gate_only",
                        "anomaly_score": score,
                        "anomaly_flag": False,
                        "anomaly_threshold": thr,
                    },
                )

        self.total_scored += len(messages)
        self.total_forwarded += forwarded
        self.forwarder.flush()
        return {"batch_size": len(messages), "forwarded": forwarded, "inference_time_ms": r.inference_time_ms}

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
                        print(
                            f"  [{EDGE_NODE_ID} {total_processed:,}] "
                            f"Gate: {stats['inference_time_ms']:.0f}ms | "
                            f"Forwarded: {self.total_forwarded}/{self.total_scored}"
                        )
            if batch_buffer:
                self.process_batch(batch_buffer)
        except Exception as e:
            print(f"\n[ERR] Pipeline error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        print(f"\n  Final Statistics ({EDGE_NODE_ID}):")
        print(f"    Total scored:    {self.total_scored:,}")
        print(f"    Total forwarded: {self.total_forwarded:,}")
        if self.forwarder:
            self.forwarder.close()
        self._shutdown_common()


class ClassifierPipeline(PipelineBase):
    """Jetson #2 role: Spark classifier on suspicious flows from the gate topic."""

    BATCH_SIZE = EDGE_BATCH_SIZE

    def __init__(self):
        print("\n" + "=" * 60)
        print(f"  IDS CLASSIFIER ({EDGE_NODE_ID})")
        print("=" * 60)
        super().__init__()

        self.spark = create_spark_session()
        self.preprocessor = FeaturePreprocessor(self.spark)
        self.engine = PredictionEngine(self.spark)

        self.consumer = KafkaConsumer(
            KAFKA_SUSPICIOUS_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_CLASSIFIER_GROUP_ID,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        print(f"[OK] Kafka Consumer subscribed to '{KAFKA_SUSPICIOUS_TOPIC}' (group: {KAFKA_CLASSIFIER_GROUP_ID})")
        print("\n" + "=" * 60)
        print("  CLASSIFIER READY - Waiting for suspicious flows...")
        print("=" * 60 + "\n")

    def process_batch(self, messages: list):
        timestamp = time.time()
        spark_df = self.preprocessor.preprocess_batch(messages)
        predictions_df, clf_stats = self.engine.predict(spark_df)

        avg_time = float(clf_stats["avg_time_ms"])
        for _ in range(clf_stats["batch_size"]):
            self.monitor.record_prediction(inference_time_ms=avg_time, is_attack=False)

        if clf_stats["attacks_found"] > 0:
            for _ in range(clf_stats["attacks_found"]):
                self.monitor.record_prediction(inference_time_ms=0, is_attack=True)
            engine_stats = self.engine.get_stats()
            self._send_alert(
                (
                    f"[ALERT] ATTACK DETECTED ({EDGE_NODE_ID})\n"
                    f"Batch attacks: {clf_stats['attacks_found']}/{clf_stats['batch_size']}\n"
                    f"Total attacks: {engine_stats['total_attacks']}\n"
                    f"Attack rate: {engine_stats['attack_rate']:.2%}\n"
                    f"Avg latency: {engine_stats['avg_inference_time_ms']:.1f}ms"
                ),
                float(clf_stats["attacks_found"]) / clf_stats["batch_size"],
            )

        if self.postgres and predictions_df is not None:
            try:
                results = predictions_df.select("prediction", "probability").collect()
                for i, (msg, row) in enumerate(zip(messages, results)):
                    pred = int(row["prediction"])
                    prob = row["probability"]
                    confidence = float(prob[int(pred)]) if prob else 0.0
                    raw_features = {
                        "route": "spark_classifier",
                        "anomaly_score": msg.get("_anomaly_score"),
                        "source_node": msg.get("_source_node"),
                    }
                    self._store_prediction(
                        timestamp=timestamp + (i * 1e-6),
                        prediction=pred,
                        confidence=confidence,
                        label="Attack" if pred == 1 else "Benign",
                        inference_time_ms=avg_time,
                        raw_features=raw_features,
                    )
            except Exception as e:
                print(f"  [WARN] DB store error: {e}")

        return clf_stats

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
                    if total_processed % 50 == 0:
                        engine_stats = self.engine.get_stats()
                        print(
                            f"  [{EDGE_NODE_ID} {total_processed:,}] "
                            f"Batch: {stats['inference_time_ms']:.0f}ms | "
                            f"Attacks: {engine_stats['total_attacks']}"
                            f"/{engine_stats['total_predictions']} | "
                            f"Avg: {engine_stats['avg_inference_time_ms']:.1f}ms"
                        )
            if batch_buffer:
                self.process_batch(batch_buffer)
        except Exception as e:
            print(f"\n[ERR] Pipeline error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        if self.engine:
            stats = self.engine.get_stats()
            print(f"\n  Final Statistics ({EDGE_NODE_ID}):")
            print(f"    Total predictions:  {stats['total_predictions']:,}")
            print(f"    Total attacks:      {stats['total_attacks']:,}")
            print(f"    Attack rate:        {stats['attack_rate']:.2%}")
            print(f"    Avg latency:        {stats['avg_inference_time_ms']:.3f} ms")
        self._shutdown_common()


def create_pipeline():
    role = EDGE_NODE_ROLE
    if role == "anomaly_gate":
        return AnomalyGatePipeline()
    if role == "classifier":
        return ClassifierPipeline()
    if role != "full":
        print(f"[WARN] Unknown EDGE_NODE_ROLE='{role}', falling back to 'full'")
    return FullPipeline()
