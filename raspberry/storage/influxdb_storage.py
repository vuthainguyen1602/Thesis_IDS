#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InfluxDB Storage \u2014 Raspberry Pi Edge IDS.

Writes system and prediction time-series metrics to InfluxDB 2.x for
real-time monitoring via Grafana dashboards.

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import time

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET


class InfluxDBStorage:
    """InfluxDB 2.x adapter for time-series metrics storage."""

    def __init__(self):
        self.client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = INFLUXDB_BUCKET
        self.org = INFLUXDB_ORG
        print(f"[OK] InfluxDB connected: {INFLUXDB_URL}")

    def write_metrics(self, metrics: dict):
        """
        Write a combined metrics dict to InfluxDB.

        Expected keys: cpu_percent, memory_percent, memory_used_mb,
                       throughput_rps, predictions_count, attacks_count,
                       avg_latency_ms, cpu_temp_celsius
        """
        now = time.time_ns()

        # System metrics point
        system_point = (
            Point("system_metrics")
            .tag("host", "raspberry-pi")
            .field("cpu_percent", float(metrics.get("cpu_percent", 0)))
            .field("memory_percent", float(metrics.get("memory_percent", 0)))
            .field("memory_used_mb", float(metrics.get("memory_used_mb", 0)))
            .field("disk_percent", float(metrics.get("disk_percent", 0)))
            .time(now, WritePrecision.NS)
        )

        if metrics.get("cpu_temp_celsius") is not None:
            system_point = system_point.field(
                "cpu_temp_celsius", float(metrics["cpu_temp_celsius"])
            )

        # Prediction metrics point
        prediction_point = (
            Point("prediction_metrics")
            .tag("host", "raspberry-pi")
            .field("throughput_rps", float(metrics.get("throughput_rps", 0)))
            .field("predictions_count", int(metrics.get("predictions_count", 0)))
            .field("attacks_count", int(metrics.get("attacks_count", 0)))
            .field("avg_latency_ms", float(metrics.get("avg_latency_ms", 0)))
            .time(now, WritePrecision.NS)
        )

        self.write_api.write(
            bucket=self.bucket,
            org=self.org,
            record=[system_point, prediction_point],
        )

    def close(self):
        """Close the InfluxDB client."""
        if self.client:
            self.client.close()
            print("[OK] InfluxDB connection closed")
