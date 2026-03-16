#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Monitor \u2014 Raspberry Pi Edge IDS.

Background thread that samples system resources (CPU, memory, disk I/O,
CPU temperature) and prediction throughput at configurable intervals.
Optionally pushes time-series data to InfluxDB for Grafana dashboards.

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import time
import threading
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METRICS_PUSH_INTERVAL


class PerformanceMonitor:
    """
    Tracks system performance and prediction throughput on the edge device.
    Designed for lightweight monitoring on Raspberry Pi.
    """

    def __init__(self, influxdb_storage=None, push_interval=None):
        self.influxdb_storage = influxdb_storage
        self.push_interval = push_interval or METRICS_PUSH_INTERVAL

        # Counters
        self._predictions_count = 0
        self._attacks_count = 0
        self._total_inference_ms = 0.0
        self._window_start = time.time()

        # Background thread
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def record_prediction(self, inference_time_ms: float, is_attack: bool):
        """Record a single prediction event."""
        with self._lock:
            self._predictions_count += 1
            self._total_inference_ms += inference_time_ms
            if is_attack:
                self._attacks_count += 1

    def get_system_metrics(self) -> dict:
        """Capture current system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # CPU temperature (Raspberry Pi specific)
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps:
                cpu_temp = temps["cpu_thermal"][0].current
        except (AttributeError, KeyError, IndexError):
            pass

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": round(memory.used / (1024 * 1024), 1),
            "memory_total_mb": round(memory.total / (1024 * 1024), 1),
            "disk_percent": disk.percent,
            "cpu_temp_celsius": cpu_temp,
        }

    def get_throughput_metrics(self) -> dict:
        """Calculate throughput metrics for the current window."""
        with self._lock:
            elapsed = time.time() - self._window_start
            throughput = self._predictions_count / elapsed if elapsed > 0 else 0
            avg_latency = (self._total_inference_ms / self._predictions_count
                          if self._predictions_count > 0 else 0)

            metrics = {
                "throughput_rps": round(throughput, 2),
                "predictions_count": self._predictions_count,
                "attacks_count": self._attacks_count,
                "avg_latency_ms": round(avg_latency, 3),
                "window_seconds": round(elapsed, 1),
            }

            # Reset window
            self._predictions_count = 0
            self._attacks_count = 0
            self._total_inference_ms = 0.0
            self._window_start = time.time()

        return metrics

    def _push_metrics_loop(self):
        """Background loop that periodically pushes metrics."""
        while self._running:
            time.sleep(self.push_interval)
            if not self._running:
                break

            system = self.get_system_metrics()
            throughput = self.get_throughput_metrics()

            combined = {**system, **throughput}

            # Print to console
            print(f"\n  [MONITOR] CPU: {system['cpu_percent']}% | "
                  f"MEM: {system['memory_percent']}% "
                  f"({system['memory_used_mb']}MB) | "
                  f"Throughput: {throughput['throughput_rps']} rps | "
                  f"Latency: {throughput['avg_latency_ms']}ms | "
                  f"Attacks: {throughput['attacks_count']}")

            if system.get("cpu_temp_celsius"):
                print(f"           Temp: {system['cpu_temp_celsius']}°C")

            # Push to InfluxDB if available
            if self.influxdb_storage:
                try:
                    self.influxdb_storage.write_metrics(combined)
                except Exception as e:
                    print(f"  [WARN] Failed to push metrics: {e}")

    def start(self):
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._push_metrics_loop, daemon=True)
        self._thread.start()
        print(f"[OK] Performance Monitor started (interval: {self.push_interval}s)")

    def stop(self):
        """Stop the background monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            print("[OK] Performance Monitor stopped")
