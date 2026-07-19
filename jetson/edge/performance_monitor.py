#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
from collections import deque

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METRICS_PUSH_INTERVAL


def _round(v, ndigits=3):
    return round(v, ndigits) if v is not None else None


def _percentile(sorted_vals, q):
    """Nearest-rank percentile on an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


# Cap the per-window raw-sample buffers so a long run cannot grow memory without
# bound; a few thousand points per window is more than enough for stable p95/p99.
_MAX_SAMPLES = 20000


class PerformanceMonitor:

    def __init__(self, influxdb_storage=None, push_interval=None, node_id="edge-node-1",
                 raw_log_path=None):
        self.influxdb_storage = influxdb_storage
        self.push_interval = push_interval or METRICS_PUSH_INTERVAL
        self.node_id = node_id
        # Raw per-flow latency log (CSV, append). Window-level percentiles pushed
        # to InfluxDB are NOT sufficient for run-level tail statistics: averaging
        # window p95s underestimates the tail. The benchmark orchestrator computes
        # the run-level p50/p95/p99 from this file instead.
        # Enable via constructor arg or env RAW_LATENCY_LOG=/path/to/file.csv
        self.raw_log_path = raw_log_path or os.getenv("RAW_LATENCY_LOG") or None
        self._raw_log_fh = None
        if self.raw_log_path:
            try:
                new_file = not os.path.exists(self.raw_log_path)
                log_dir = os.path.dirname(self.raw_log_path)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                self._raw_log_fh = open(self.raw_log_path, "a", buffering=1)
                if new_file:
                    self._raw_log_fh.write("ts_unix,node_id,inference_ms,e2e_ms\n")
                print(f"[OK] Raw latency log: {self.raw_log_path}")
            except OSError as e:
                print(f"  [WARN] Cannot open raw latency log {self.raw_log_path}: {e}")
                self._raw_log_fh = None

        self._predictions_count = 0
        self._attacks_count = 0
        self._total_inference_ms = 0.0
        # Raw per-sample latency buffers for the current push window. Percentiles
        # (p50/p95/p99) MUST be computed from these raw distributions — taking a
        # percentile of already-averaged values (e.g. per-host means) is invalid.
        self._inf_latencies = deque(maxlen=_MAX_SAMPLES)   # preprocess+predict (ms)
        self._e2e_latencies = deque(maxlen=_MAX_SAMPLES)    # send -> verdict (ms)
        self._window_start = time.time()

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def record_prediction(self, inference_time_ms: float, is_attack: bool,
                          end_to_end_ms: float = None):
        """Record one verdict.

        ``inference_time_ms`` is the on-node preprocess+predict cost. When the
        producer stamps a send time (``_timestamp``) into the message, pass
        ``end_to_end_ms`` (now - send_time) so the true end-to-end latency
        distribution — Kafka consume + queueing + inference + result write — is
        captured, not just inference.
        """
        with self._lock:
            self._predictions_count += 1
            self._total_inference_ms += inference_time_ms
            if inference_time_ms > 0:
                self._inf_latencies.append(inference_time_ms)
            if end_to_end_ms is not None and end_to_end_ms >= 0:
                self._e2e_latencies.append(end_to_end_ms)
            if is_attack:
                self._attacks_count += 1
            if self._raw_log_fh is not None:
                e2e_str = f"{end_to_end_ms:.3f}" if (end_to_end_ms is not None
                                                     and end_to_end_ms >= 0) else ""
                self._raw_log_fh.write(
                    f"{time.time():.3f},{self.node_id},{inference_time_ms:.3f},{e2e_str}\n")

    def get_system_metrics(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            for key in ("cpu_thermal", "soc_thermal", "gpu-thermal", "thermal-fan-est", "tj-thermal"):
                if key in temps and temps[key] and temps[key][0].current is not None:
                    cpu_temp = temps[key][0].current
                    break
        except Exception:
            # psutil can raise TypeError/OSError on some Jetson kernels when a
            # thermal zone reports a null value; temperature is non-critical.
            pass
        if cpu_temp is None:
            # Fall back to the Tegra thermal zone read directly.
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as _f:
                    cpu_temp = round(float(_f.read().strip()) / 1000.0, 1)
            except Exception:
                cpu_temp = None

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": round(memory.used / (1024 * 1024), 1),
            "memory_total_mb": round(memory.total / (1024 * 1024), 1),
            "disk_percent": disk.percent,
            "cpu_temp_celsius": cpu_temp,
        }

    def get_throughput_metrics(self) -> dict:
        with self._lock:
            elapsed = time.time() - self._window_start
            throughput = self._predictions_count / elapsed if elapsed > 0 else 0
            avg_latency = (self._total_inference_ms / self._predictions_count
                          if self._predictions_count > 0 else 0)

            inf_sorted = sorted(self._inf_latencies)
            e2e_sorted = sorted(self._e2e_latencies)

            metrics = {
                "throughput_rps": round(throughput, 2),
                "predictions_count": self._predictions_count,
                "attacks_count": self._attacks_count,
                "avg_latency_ms": round(avg_latency, 3),
                # Inference-latency percentiles from raw samples.
                "latency_p50_ms": _round(_percentile(inf_sorted, 0.50)),
                "latency_p95_ms": _round(_percentile(inf_sorted, 0.95)),
                "latency_p99_ms": _round(_percentile(inf_sorted, 0.99)),
                # End-to-end (send -> verdict) percentiles, populated only when
                # the producer stamps a send time into the message.
                "e2e_latency_avg_ms": _round(
                    sum(e2e_sorted) / len(e2e_sorted) if e2e_sorted else None),
                "e2e_latency_p50_ms": _round(_percentile(e2e_sorted, 0.50)),
                "e2e_latency_p95_ms": _round(_percentile(e2e_sorted, 0.95)),
                "e2e_latency_p99_ms": _round(_percentile(e2e_sorted, 0.99)),
                "window_seconds": round(elapsed, 1),
            }

            self._predictions_count = 0
            self._attacks_count = 0
            self._total_inference_ms = 0.0
            self._inf_latencies.clear()
            self._e2e_latencies.clear()
            self._window_start = time.time()

        return metrics

    def _push_metrics_loop(self):
        while self._running:
            time.sleep(self.push_interval)
            if not self._running:
                break

            system = self.get_system_metrics()
            throughput = self.get_throughput_metrics()

            combined = {**system, **throughput}

            print(f"\n  [MONITOR:{self.node_id}] CPU: {system['cpu_percent']}% | "
                  f"MEM: {system['memory_percent']}% "
                  f"({system['memory_used_mb']}MB) | "
                  f"Throughput: {throughput['throughput_rps']} rps | "
                  f"Latency: {throughput['avg_latency_ms']}ms | "
                  f"Attacks: {throughput['attacks_count']}")

            if system.get("cpu_temp_celsius"):
                print(f"           Temp: {system['cpu_temp_celsius']} C")

            if self.influxdb_storage:
                try:
                    self.influxdb_storage.write_metrics(combined)
                except Exception as e:
                    print(f"  [WARN] Failed to push metrics: {e}")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._push_metrics_loop, daemon=True)
        self._thread.start()
        print(f"[OK] Performance Monitor started (interval: {self.push_interval}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            print("[OK] Performance Monitor stopped")
        if self._raw_log_fh is not None:
            try:
                self._raw_log_fh.close()
            except OSError:
                pass
            self._raw_log_fh = None
