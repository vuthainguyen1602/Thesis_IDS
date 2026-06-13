#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EDGE_NODE_ID, ALERT_ENABLED, ALERT_COOLDOWN

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

from edge.performance_monitor import PerformanceMonitor


class PipelineBase:

    def __init__(self):
        self.running = True
        self.last_alert_time = 0
        self.postgres = None
        self.influxdb = None
        self.alerting = None
        self.monitor = None
        self.consumer = None
        self.spark = None

        print(f"[INFO] Node ID: {EDGE_NODE_ID}")

        if PostgresStorage:
            try:
                self.postgres = PostgresStorage()
                self.postgres.init_tables()
            except Exception as e:
                print(f"[WARN] PostgreSQL disabled: {e}")

        if InfluxDBStorage:
            try:
                self.influxdb = InfluxDBStorage(node_id=EDGE_NODE_ID)
            except Exception as e:
                print(f"[WARN] InfluxDB disabled: {e}")

        if AlertSystem and ALERT_ENABLED:
            try:
                self.alerting = AlertSystem(node_id=EDGE_NODE_ID)
            except Exception as e:
                print(f"[WARN] Alert System disabled: {e}")
        elif not ALERT_ENABLED:
            print("[INFO] Alerts disabled on this node (ALERT_ENABLED=0)")

        self.monitor = PerformanceMonitor(
            influxdb_storage=self.influxdb,
            node_id=EDGE_NODE_ID,
        )
        self.monitor.start()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n[WARN] Shutdown signal received...")
        self.running = False

    def _node_meta(self, raw_features=None):
        meta = {"node_id": EDGE_NODE_ID}
        if raw_features:
            meta.update(raw_features)
        return meta

    def _store_prediction(self, **kwargs):
        if not self.postgres:
            return
        raw_features = kwargs.pop("raw_features", None)
        kwargs["raw_features"] = self._node_meta(raw_features)
        kwargs["node_id"] = EDGE_NODE_ID
        self.postgres.store_prediction(**kwargs)

    def _send_alert(self, message: str, confidence: float):
        now = time.time()
        if now - self.last_alert_time <= ALERT_COOLDOWN:
            return
        self.last_alert_time = now

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
                    confidence=confidence,
                    node_id=EDGE_NODE_ID,
                )
            except Exception:
                pass

    def _shutdown_common(self):
        print("\n" + "=" * 60)
        print(f"  SHUTTING DOWN IDS EDGE ({EDGE_NODE_ID})")
        print("=" * 60)

        if self.monitor:
            self.monitor.stop()
        if self.consumer:
            self.consumer.close()
        if self.postgres:
            self.postgres.close()
        if self.influxdb:
            self.influxdb.close()
        if self.spark:
            self.spark.stop()

        print("\n[OK] Pipeline shutdown complete.")
