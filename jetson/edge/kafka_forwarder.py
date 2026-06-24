#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time

from kafka import KafkaProducer

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_SUSPICIOUS_TOPIC


class SuspiciousFlowForwarder:

    def __init__(self, bootstrap_servers=None, topic=None):
        self.topic = topic or KAFKA_SUSPICIOUS_TOPIC
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks=1,
            retries=3,
            linger_ms=5,
        )
        self.total_forwarded = 0
        print(f"[OK] Suspicious flow forwarder -> topic '{self.topic}'")

    def forward(self, message: dict, *, anomaly_score: float, source_node: str) -> None:
        payload = dict(message)
        payload["_forwarded_at"] = time.time()
        payload["_anomaly_score"] = float(anomaly_score)
        payload["_source_node"] = source_node
        key = str(int(time.time() * 1000))
        self.producer.send(self.topic, key=key, value=payload)
        self.total_forwarded += 1

    def flush(self):
        self.producer.flush()

    def close(self):
        self.flush()
        self.producer.close()
        print(f"[OK] Forwarder closed ({self.total_forwarded:,} messages sent)")
