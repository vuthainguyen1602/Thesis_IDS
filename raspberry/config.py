#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ids-network-flow")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ids-edge-consumer")

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model", "ids_pipeline_model"))
FEATURES_PATH = os.getenv("FEATURES_PATH", os.path.join(os.path.dirname(__file__), "model", "feature_columns.json"))

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
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "ids-alert@raspberry-pi.local")

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", "")
SEND_RATE = int(os.getenv("SEND_RATE", "100"))

METRICS_PUSH_INTERVAL = int(os.getenv("METRICS_PUSH_INTERVAL", "10"))
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "60"))

SHAP_TOP_FEATURES = [
    "flow_duration", "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min", "fwd_packet_length_mean",
    "bwd_packet_length_max", "bwd_packet_length_mean", "bwd_packet_length_std",
    "flow_bytes_s", "flow_packets_s", "flow_iat_mean", "flow_iat_std",
    "flow_iat_max", "flow_iat_min", "fwd_iat_total", "fwd_iat_mean",
    "bwd_iat_total", "bwd_iat_mean", "fwd_psh_flags", "bwd_packets_s",
    "min_packet_length", "max_packet_length", "packet_length_mean",
    "packet_length_std", "packet_length_variance", "average_packet_size",
    "destination_port",
]
