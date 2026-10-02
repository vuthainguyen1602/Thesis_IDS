#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ids-network-flow")
KAFKA_SUSPICIOUS_TOPIC = os.getenv("KAFKA_SUSPICIOUS_TOPIC", "ids-suspicious-flow")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ids-edge-consumer")
KAFKA_CLASSIFIER_GROUP_ID = os.getenv("KAFKA_CLASSIFIER_GROUP_ID", "ids-classifier-consumer")

# Distributed edge deployment (Jetson Orin Nano Super cluster)
# Roles: full | anomaly_gate | classifier
#   full          - complete pipeline on each node (Kafka consumer group scales horizontally)
#   anomaly_gate  - Jetson #1: anomaly filter, forward suspicious flows to KAFKA_SUSPICIOUS_TOPIC
#   classifier    - Jetson #2: Spark classifier on suspicious flows only
EDGE_NODE_ID = os.getenv("EDGE_NODE_ID", "edge-node-1")
EDGE_NODE_ROLE = os.getenv("EDGE_NODE_ROLE", "full").strip().lower()
ALERT_ENABLED = os.getenv("ALERT_ENABLED", "1").strip() in ("1", "true", "True", "yes", "YES")

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model", "ids_pipeline_model"))
FEATURES_PATH = os.getenv("FEATURES_PATH", os.path.join(os.path.dirname(__file__), "model", "feature_columns.json"))

# Classifier-tier backend: "spark" reproduces the published deployment;
# "numpy" and "onnx" serve exported artifacts without a JVM on the board.
EDGE_ENGINE = os.getenv("EDGE_ENGINE", "spark").strip().lower()
NUMPY_MODEL_PATH = os.getenv(
    "NUMPY_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model", "ids_rf_numpy.npz"),
)
ONNX_MODEL_PATH = os.getenv(
    "ONNX_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model", "ids_rf.onnx"),
)
ONNX_PROVIDERS = [p.strip() for p in os.getenv(
    "ONNX_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]
ONNX_INTRA_THREADS = int(os.getenv("ONNX_INTRA_THREADS", "0"))  # 0 = onnxruntime default

ANOMALY_ENABLED = os.getenv("ANOMALY_ENABLED", "0").strip() in ("1", "true", "True", "yes", "YES")
ANOMALY_MODEL_PATH = os.getenv(
    "ANOMALY_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model", "anomaly_autoencoder.pkl"),
)
ANOMALY_SCALER_PATH = os.getenv(
    "ANOMALY_SCALER_PATH",
    os.path.join(os.path.dirname(__file__), "model", "anomaly_scaler.pkl"),
)
ANOMALY_THRESHOLD_PATH = os.getenv(
    "ANOMALY_THRESHOLD_PATH",
    os.path.join(os.path.dirname(__file__), "model", "anomaly_threshold.json"),
)
# The gate is an unsupervised anomaly detector, so it scores all leak-free
# numeric features; the classifier uses its own SHAP subset (FEATURES_PATH).
ANOMALY_FEATURES_PATH = os.getenv(
    "ANOMALY_FEATURES_PATH",
    os.path.join(os.path.dirname(__file__), "model", "anomaly_feature_columns.json"),
)

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "ids_edge")
POSTGRES_USER = os.getenv("POSTGRES_USER", "ids")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "ids_password")

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "ids-edge-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "thesis")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "ids_metrics")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SMTP_HOST = os.getenv("SMTP_HOST", "sandbox.smtp.mailtrap.io")
SMTP_PORT = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "ids-alert@jetson-pi.local")

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", "")
SEND_RATE = int(os.getenv("SEND_RATE", "100"))
EDGE_BATCH_SIZE = int(os.getenv("EDGE_BATCH_SIZE", "20"))
# A batch is processed when it is full OR its oldest flow has waited this long.
# Without the time limit a tier that sees few flows (the classifier behind the
# gate gets ~11% of traffic) can hold a partial batch of attacks indefinitely.
EDGE_BATCH_MAX_WAIT_S = float(os.getenv("EDGE_BATCH_MAX_WAIT_S", "1.0"))
SPARK_MASTER = os.getenv("SPARK_MASTER", "")
if not SPARK_MASTER:
    SPARK_MASTER = os.getenv("SPARK_MASTER_URL", "")
# Edge / ML: set SPARK_MASTER=spark://<MAC_IP>:7077 in .env (see cluster/spark_cluster.env.example)
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "512m")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "512m")
SPARK_SHUFFLE_PARTITIONS = os.getenv("SPARK_SHUFFLE_PARTITIONS", "2")
SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", f"IDS_Edge_{EDGE_NODE_ID}")
SPARK_WORKER_MEMORY = os.getenv("SPARK_WORKER_MEMORY", "5g")
SPARK_WORKER_CORES = os.getenv("SPARK_WORKER_CORES", "2")

METRICS_PUSH_INTERVAL = int(os.getenv("METRICS_PUSH_INTERVAL", "10"))
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "60"))

# Fallback copy of the leakage-free SHAP Top-30 (results/ml_06_feature_selection_shap/
# shap_feature_importance.csv, port columns excluded before ranking). The exported
# model's own list lives in model/feature_columns.json and wins when present;
# the export scripts read the CSV directly, so this copy only matters on a board
# that has no feature_columns.json.
SHAP_TOP_FEATURES = [
    "bwd_packet_length_std",
    "init_win_bytes_backward",
    "init_win_bytes_forward",
    "bwd_packet_length_mean",
    "fwd_packet_length_max",
    "average_packet_size",
    "bwd_packets_s",
    "bwd_packet_length_min",
    "idle_min",
    "packet_length_std",
    "fwd_header_length34",
    "total_fwd_packets",
    "total_length_of_bwd_packets",
    "fin_flag_count",
    "min_seg_size_forward",
    "fwd_iat_min",
    "flow_iat_min",
    "fwd_packet_length_mean",
    "bwd_packet_length_max",
    "fwd_iat_std",
    "avg_bwd_segment_size",
    "total_backward_packets",
    "act_data_pkt_fwd",
    "idle_max",
    "total_length_of_fwd_packets",
    "flow_iat_max",
    "active_min",
    "fwd_iat_mean",
    "psh_flag_count",
    "fwd_iat_max",
]
