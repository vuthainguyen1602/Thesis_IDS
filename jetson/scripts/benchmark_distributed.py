#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distributed edge IDS benchmark orchestrator for SOICT 2026.

Usage examples (run from jetson/ with venv activated):

  # Local inference micro-benchmark on one node:
  python scripts/benchmark_distributed.py local --samples 500

  # Send Kafka traffic from PC for 60s at 100 flows/s:
  python scripts/benchmark_distributed.py send --duration 60 --rate 100

  # Collect metrics from InfluxDB + Postgres for the last 5 minutes:
  python scripts/benchmark_distributed.py collect --window-minutes 5

  # Full end-to-end: warmup, send, collect (requires edge pipelines running):
  python scripts/benchmark_distributed.py run --mode split --duration 60 --rate 100

  # Merge multiple JSON result files into paper CSV:
  python scripts/benchmark_distributed.py merge --input results/*.json --output-csv ../papers/soict2026/results/benchmarks/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_CSV_PATH,
    EDGE_NODE_ID,
    EDGE_NODE_ROLE,
    INFLUXDB_BUCKET,
    INFLUXDB_ORG,
    INFLUXDB_TOKEN,
    INFLUXDB_URL,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


def detect_device_name() -> str:
    env = os.getenv("EDGE_DEVICE_NAME")
    if env:
        return env
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip("\x00\n ")
            if model:
                return model
    except OSError:
        pass
    return "unknown-device"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[idx]


def epoch_to_rfc3339(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_run_percentiles_from_raw(raw_glob: str, start_epoch: float | None,
                                     end_epoch: float | None) -> dict[str, dict]:
    """Run-level latency percentiles from the raw per-flow CSV logs written by
    PerformanceMonitor (RAW_LATENCY_LOG). This is the statistically valid way to
    report p50/p95/p99 for a run: percentiles over the FULL raw distribution
    restricted to the load window — never averages of window-level percentiles.

    Returns {node_id: {n, inference: {p50,p95,p99,mean}, e2e: {...}}}.
    """
    per_node_inf: dict[str, list[float]] = {}
    per_node_e2e: dict[str, list[float]] = {}
    paths = sorted(glob.glob(raw_glob))
    if not paths:
        print(f"[WARN] No raw latency logs match: {raw_glob}")
        return {}

    for path in paths:
        with open(path, "r") as f:
            header = f.readline()
            if not header.startswith("ts_unix"):
                f.seek(0)
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 3:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                if start_epoch is not None and ts < start_epoch:
                    continue
                if end_epoch is not None and ts > end_epoch:
                    continue
                node = parts[1] or "unknown"
                try:
                    inf_ms = float(parts[2])
                except ValueError:
                    inf_ms = None
                if inf_ms is not None and inf_ms > 0:
                    per_node_inf.setdefault(node, []).append(inf_ms)
                if len(parts) >= 4 and parts[3]:
                    try:
                        per_node_e2e.setdefault(node, []).append(float(parts[3]))
                    except ValueError:
                        pass

    def _stats(vals: list[float]) -> dict:
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 3),
            "p50": round(percentile(vals, 0.50), 3),
            "p95": round(percentile(vals, 0.95), 3),
            "p99": round(percentile(vals, 0.99), 3),
        }

    out: dict[str, dict] = {}
    for node in sorted(set(per_node_inf) | set(per_node_e2e)):
        out[node] = {}
        if per_node_inf.get(node):
            out[node]["inference"] = _stats(per_node_inf[node])
        if per_node_e2e.get(node):
            out[node]["e2e"] = _stats(per_node_e2e[node])
    return out


def run_local_benchmark(samples: int, batch_size: int) -> dict:
    """Delegate to benchmark.py and enrich with node metadata."""
    script = os.path.join(os.path.dirname(__file__), "benchmark.py")
    cmd = [sys.executable, script, "--samples", str(samples), "--batch-size", str(batch_size)]
    subprocess.run(cmd, check=True)

    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark_results.json")
    with open(results_path, "r") as f:
        payload = json.load(f)

    payload.update({
        "benchmark_type": "local",
        "node_id": EDGE_NODE_ID,
        "mode": EDGE_NODE_ROLE,
        "device": detect_device_name(),
        "timestamp": utc_now_iso(),
    })
    return payload


def send_kafka_traffic(duration_s: int, rate: int, csv_path: str | None, topic: str) -> dict:
    """Send CSV rows to Kafka for a fixed duration (subprocess with timeout)."""
    csv_path = csv_path or DATA_CSV_PATH
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}. Set DATA_CSV_PATH in .env")

    sender = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sender", "data_sender.py")
    cmd = [
        sys.executable,
        sender,
        "--csv", csv_path,
        "--rate", str(rate),
        "--broker", KAFKA_BOOTSTRAP_SERVERS,
        "--topic", topic,
    ]

    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        proc.wait(timeout=duration_s)
        interrupted = False
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=5)
        interrupted = True

    elapsed = time.time() - start
    return {
        "benchmark_type": "send",
        "duration_s": duration_s,
        "requested_rate_rps": rate,
        "elapsed_s": round(elapsed, 2),
        "interrupted_after_timeout": interrupted,
        "csv_path": csv_path,
        "topic": topic,
        "broker": KAFKA_BOOTSTRAP_SERVERS,
        "timestamp": utc_now_iso(),
    }


def _flux_range(window_minutes: int, start_epoch: float | None,
                end_epoch: float | None) -> str:
    """Range clause aligned to the load window when available. Aligning matters:
    a trailing -Nm window mixes idle/warmup samples into the run statistics."""
    if start_epoch is not None and end_epoch is not None:
        return (f"range(start: {epoch_to_rfc3339(start_epoch)}, "
                f"stop: {epoch_to_rfc3339(end_epoch)})")
    return f"range(start: -{window_minutes}m)"


def query_influx_metrics(window_minutes: int, start_epoch: float | None = None,
                         end_epoch: float | None = None) -> dict[str, dict]:
    """Per-host mean of window-level metrics from InfluxDB.

    NOTE: window-level p95 values averaged over windows are NOT a run-level p95
    (they smooth the tail). They are kept for dashboarding only; the run-level
    percentiles used in the paper come from compute_run_percentiles_from_raw().
    """
    try:
        from influxdb_client import InfluxDBClient
    except ImportError:
        print("[WARN] influxdb-client not installed; skipping InfluxDB query")
        return {}

    flux = f"""
from(bucket: "{INFLUXDB_BUCKET}")
  |> {_flux_range(window_minutes, start_epoch, end_epoch)}
  |> filter(fn: (r) => r._measurement == "prediction_metrics")
  |> filter(fn: (r) => r._field == "throughput_rps" or r._field == "avg_latency_ms" or r._field == "latency_p50_ms" or r._field == "latency_p95_ms" or r._field == "latency_p99_ms" or r._field == "e2e_latency_avg_ms" or r._field == "e2e_latency_p95_ms")
  |> group(columns: ["host", "_field"])
  |> mean()
"""
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    try:
        tables = client.query_api().query(flux)
    finally:
        client.close()

    per_host: dict[str, dict] = {}
    for table in tables:
        for record in table.records:
            host = record.values.get("host", "unknown")
            field = record.get_field()
            value = float(record.get_value())
            per_host.setdefault(host, {})
            per_host[host][field] = round(value, 3)

    return per_host


def query_postgres_final_throughput(window_minutes: int,
                                    start_epoch: float | None = None,
                                    end_epoch: float | None = None) -> dict:
    """Pipeline throughput = FINAL verdicts per second within the load window.

    Summing per-node throughput_rps double-counts flows in pipeline-split mode
    (the same flow passes gate then classifier). Every flow gets exactly one row
    in ``predictions`` at the stage that issued its final verdict (gate-skip ->
    benign at the gate node; forwarded -> classifier verdict), so counting rows
    per second is the correct pipeline-level throughput for every mode.
    """
    try:
        import psycopg2
    except ImportError:
        print("[WARN] psycopg2 not installed; skipping Postgres throughput query")
        return {}

    end = end_epoch if end_epoch is not None else time.time()
    start = start_epoch if start_epoch is not None else end - window_minutes * 60
    duration = max(end - start, 1e-9)
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM predictions WHERE timestamp >= %s AND timestamp <= %s",
                (start, end),
            )
            total = int(cur.fetchone()[0] or 0)
    finally:
        conn.close()

    return {
        "final_verdicts": total,
        "window_s": round(duration, 1),
        "throughput_final_rps": round(total / duration, 2),
    }


def query_postgres_gate_skip(window_minutes: int,
                             start_epoch: float | None = None,
                             end_epoch: float | None = None) -> dict:
    """Estimate gate skip ratio from stored predictions."""
    try:
        import psycopg2
    except ImportError:
        print("[WARN] psycopg2 not installed; skipping Postgres query")
        return {}

    since = start_epoch if start_epoch is not None else time.time() - window_minutes * 60
    until = end_epoch if end_epoch is not None else time.time()
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE raw_features->>'route' = 'anomaly_gate_only'
                    ) AS benign_skipped,
                    COUNT(*) FILTER (
                        WHERE raw_features->>'route' = 'anomaly_gate_only'
                           OR raw_features->>'route' IS NULL
                           OR prediction = 1
                    ) AS total_gate_events
                FROM predictions
                WHERE timestamp >= %s AND timestamp <= %s
                """,
                (since, until),
            )
            row = cur.fetchone()
            benign_skipped = int(row[0] or 0)
            total = int(row[1] or 0)
    finally:
        conn.close()

    skip_pct = round(100.0 * benign_skipped / total, 2) if total > 0 else None
    return {
        "benign_skipped": benign_skipped,
        "total_gate_events": total,
        "gate_skip_pct": skip_pct,
    }


def query_system_metrics(window_minutes: int, start_epoch: float | None = None,
                         end_epoch: float | None = None) -> dict[str, dict]:
    try:
        from influxdb_client import InfluxDBClient
    except ImportError:
        return {}

    flux = f"""
from(bucket: "{INFLUXDB_BUCKET}")
  |> {_flux_range(window_minutes, start_epoch, end_epoch)}
  |> filter(fn: (r) => r._measurement == "system_metrics")
  |> filter(fn: (r) => r._field == "cpu_percent" or r._field == "memory_percent" or r._field == "cpu_temp_celsius")
  |> group(columns: ["host", "_field"])
  |> mean()
"""
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    try:
        tables = client.query_api().query(flux)
    finally:
        client.close()

    per_host: dict[str, dict] = {}
    for table in tables:
        for record in table.records:
            host = record.values.get("host", "unknown")
            field = record.get_field()
            value = float(record.get_value())
            per_host.setdefault(host, {})
            per_host[host][field] = round(value, 2)

    return per_host


def collect_metrics(window_minutes: int, deploy_mode: str,
                    start_epoch: float | None = None,
                    end_epoch: float | None = None,
                    raw_latency_glob: str | None = None) -> dict:
    influx_pred = query_influx_metrics(window_minutes, start_epoch, end_epoch)
    influx_sys = query_system_metrics(window_minutes, start_epoch, end_epoch)
    gate = query_postgres_gate_skip(window_minutes, start_epoch, end_epoch)
    final_tp = query_postgres_final_throughput(window_minutes, start_epoch, end_epoch)
    raw_stats = (compute_run_percentiles_from_raw(raw_latency_glob, start_epoch, end_epoch)
                 if raw_latency_glob else {})

    hosts = sorted(set(influx_pred) | set(influx_sys))
    nodes = []
    throughputs = []
    latencies = []

    node_p95s = []
    e2e_p95s = []
    e2e_avgs = []
    for host in hosts:
        pred = influx_pred.get(host, {})
        sys_m = influx_sys.get(host, {})
        tp = pred.get("throughput_rps", 0.0)
        lat = pred.get("avg_latency_ms", 0.0)
        if tp > 0:
            throughputs.append(tp)
        if lat > 0:
            latencies.append(lat)
        if pred.get("latency_p95_ms"):
            node_p95s.append(pred["latency_p95_ms"])
        if pred.get("e2e_latency_p95_ms"):
            e2e_p95s.append(pred["e2e_latency_p95_ms"])
        if pred.get("e2e_latency_avg_ms"):
            e2e_avgs.append(pred["e2e_latency_avg_ms"])
        nodes.append({
            "node_id": host,
            "throughput_rps": tp,
            "avg_latency_ms": lat,
            "latency_p50_ms": pred.get("latency_p50_ms"),
            "latency_p95_ms": pred.get("latency_p95_ms"),
            "latency_p99_ms": pred.get("latency_p99_ms"),
            "e2e_latency_avg_ms": pred.get("e2e_latency_avg_ms"),
            "e2e_latency_p95_ms": pred.get("e2e_latency_p95_ms"),
            "cpu_percent": sys_m.get("cpu_percent"),
            "memory_percent": sys_m.get("memory_percent"),
            "cpu_temp_celsius": sys_m.get("cpu_temp_celsius"),
        })

    # --- Run-level latency (preferred: raw per-flow logs) -------------------
    # Percentiles MUST come from the full raw distribution of the load window.
    # InfluxDB only stores window-level p95s; their mean/max is a fallback
    # approximation, and is labelled as such so it never silently enters the
    # paper table.
    run_inf_p95 = None
    run_e2e_p95 = None
    latency_source = None
    if raw_stats:
        inf_p95s = [v["inference"]["p95"] for v in raw_stats.values() if "inference" in v]
        e2e_p95s_raw = [v["e2e"]["p95"] for v in raw_stats.values() if "e2e" in v]
        run_inf_p95 = round(max(inf_p95s), 3) if inf_p95s else None
        run_e2e_p95 = round(max(e2e_p95s_raw), 3) if e2e_p95s_raw else None
        latency_source = "raw_per_flow_logs"
    elif node_p95s:
        run_inf_p95 = round(max(node_p95s), 2)
        run_e2e_p95 = round(max(e2e_p95s), 2) if e2e_p95s else None
        latency_source = "influx_window_p95_APPROXIMATION_DO_NOT_PUBLISH"
        print("[WARN] No raw latency logs supplied (--raw-latency-glob). "
              "Falling back to mean-of-window p95 from InfluxDB — this "
              "underestimates the tail and must NOT be used in the paper.")

    # --- Pipeline throughput -------------------------------------------------
    # In split mode, summing per-node rates double-counts forwarded flows; the
    # correct figure is final verdicts/s from Postgres. Per-node rates are kept
    # for diagnosis only.
    aggregate = {
        "throughput_final_rps": final_tp.get("throughput_final_rps"),
        "throughput_rps_per_node_sum_DIAGNOSTIC_ONLY": (
            round(sum(throughputs), 2) if throughputs else None),
        "latency_avg_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_p95_ms": run_inf_p95,
        "e2e_latency_avg_ms": round(statistics.mean(e2e_avgs), 2) if e2e_avgs else None,
        "e2e_latency_p95_ms": run_e2e_p95,
        "latency_source": latency_source,
        "latency_note": ("run-level p95 = worst node's percentile over its RAW "
                         "per-flow samples within [load_start, load_end]; "
                         "e2e = send->verdict, assumes NTP-synced clocks"),
    }

    return {
        "benchmark_type": "collect",
        "deploy_mode": deploy_mode,
        "window_minutes": window_minutes,
        "load_start_epoch": start_epoch,
        "load_end_epoch": end_epoch,
        "timestamp": utc_now_iso(),
        "gate_skip": gate,
        "final_throughput": final_tp,
        "raw_latency_per_node": raw_stats,
        "nodes": nodes,
        "aggregate": aggregate,
        "device": detect_device_name(),
        "local_cpu_percent": round(psutil.cpu_percent(interval=0.5), 1),
        "local_memory_percent": round(psutil.virtual_memory().percent, 1),
    }


def run_end_to_end(
    deploy_mode: str,
    duration_s: int,
    rate: int,
    warmup_s: int,
    window_minutes: int,
    csv_path: str | None,
    raw_latency_glob: str | None = None,
) -> dict:
    # Warmup MUST be representative load, not an idle sleep: JVM/JIT and Spark
    # code paths only warm up under traffic. Warmup samples are excluded from
    # the measured window below (they fall before load_start).
    warmup_info = None
    if warmup_s > 0:
        print(f"\n[WARMUP] Sending {warmup_s}s of warmup traffic at {rate} flows/s "
              f"(excluded from measurement)...")
        warmup_info = send_kafka_traffic(warmup_s, rate, csv_path, KAFKA_TOPIC)
        time.sleep(2)  # small gap so warmup flows drain out of the queues

    load_start = time.time()
    send_info = send_kafka_traffic(duration_s, rate, csv_path, KAFKA_TOPIC)
    load_end = time.time()
    print(f"\n[WAIT] Draining metrics (load window "
          f"{epoch_to_rfc3339(load_start)} .. {epoch_to_rfc3339(load_end)})...")
    time.sleep(min(30, window_minutes * 10))

    collect_info = collect_metrics(
        window_minutes, deploy_mode,
        start_epoch=load_start, end_epoch=load_end,
        raw_latency_glob=raw_latency_glob,
    )
    return {
        "benchmark_type": "run",
        "deploy_mode": deploy_mode,
        "warmup": warmup_info,
        "send": send_info,
        "load_start_epoch": load_start,
        "load_end_epoch": load_end,
        "collect": collect_info,
        "timestamp": utc_now_iso(),
        "node_id": EDGE_NODE_ID,
        "role": EDGE_NODE_ROLE,
    }


def save_json(payload: dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Wrote {output_path}")
    return output_path


def row_from_result(data: dict) -> dict:
    """Flatten one benchmark JSON into a CSV row for the paper table."""
    mode = data.get("deploy_mode") or data.get("mode") or "unknown"
    agg = data.get("aggregate") or {}
    collect = data.get("collect") or {}
    if collect and "aggregate" in collect:
        agg = collect["aggregate"]
        mode = collect.get("deploy_mode", mode)
    gate = (collect.get("gate_skip") if collect else data.get("gate_skip")) or {}

    return {
        "mode": mode,
        "node_id": data.get("node_id", EDGE_NODE_ID),
        # Pipeline throughput = final verdicts/s (Postgres). Never the per-node
        # sum, which double-counts forwarded flows in split mode.
        "throughput_rps": agg.get("throughput_final_rps") or data.get("throughput_rps"),
        "latency_p95_ms": agg.get("latency_p95_ms") or data.get("latency_batch_p95_ms"),
        "latency_avg_ms": agg.get("latency_avg_ms") or data.get("latency_per_sample_ms"),
        "e2e_latency_p95_ms": agg.get("e2e_latency_p95_ms"),
        "latency_source": agg.get("latency_source"),
        "gate_skip_pct": gate.get("gate_skip_pct"),
        "attack_f1": data.get("attack_f1"),
        "cpu_percent": data.get("avg_cpu_percent"),
        "memory_percent": data.get("avg_memory_percent"),
        "device": data.get("device", detect_device_name()),
        "timestamp": data.get("timestamp", utc_now_iso()),
    }


def merge_results(input_glob: str, output_csv: str) -> None:
    paths = sorted(glob.glob(input_glob))
    if not paths:
        print(f"[ERR] No files match: {input_glob}")
        sys.exit(1)

    rows = []
    for path in paths:
        with open(path, "r") as f:
            rows.append(row_from_result(json.load(f)))

    fieldnames = [
        "mode", "node_id", "throughput_rps", "latency_p95_ms", "latency_avg_ms",
        "e2e_latency_p95_ms", "latency_source",
        "gate_skip_pct", "attack_f1", "cpu_percent", "memory_percent", "device", "timestamp",
    ]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Merged {len(rows)} runs -> {output_csv}")

    # Warn loudly if any run's tail latency came from the InfluxDB fallback.
    bad = [r for r in rows if r.get("latency_source")
           and "APPROXIMATION" in str(r["latency_source"])]
    if bad:
        print(f"[WARN] {len(bad)} run(s) lack raw latency logs — their p95 is an "
              "approximation and must not be published. Re-run with RAW_LATENCY_LOG "
              "set on each node and pass --raw-latency-glob.")

    # Repeated runs per mode -> mean ± std for the paper table.
    per_mode: dict[str, list[dict]] = {}
    for r in rows:
        per_mode.setdefault(str(r["mode"]), []).append(r)
    summary_rows = []
    for mode, mode_rows in sorted(per_mode.items()):
        def _vals(key):
            return [float(r[key]) for r in mode_rows if r.get(key) is not None]
        def _mean_std(key):
            v = _vals(key)
            if not v:
                return None, None
            return (round(statistics.mean(v), 2),
                    round(statistics.stdev(v), 2) if len(v) > 1 else 0.0)
        tp_m, tp_s = _mean_std("throughput_rps")
        p95_m, p95_s = _mean_std("latency_p95_ms")
        summary_rows.append({
            "mode": mode, "n_runs": len(mode_rows),
            "throughput_rps_mean": tp_m, "throughput_rps_std": tp_s,
            "latency_p95_ms_mean": p95_m, "latency_p95_ms_std": p95_s,
        })
    stats_csv = output_csv.replace(".csv", "_mean_std.csv")
    with open(stats_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[OK] Per-mode mean±std ({len(summary_rows)} modes) -> {stats_csv}")
    for s in summary_rows:
        if s["n_runs"] < 3:
            print(f"[WARN] mode '{s['mode']}' has only {s['n_runs']} run(s); "
                  "use >=3 (ideally 5) repeats before reporting mean±std.")

    print_latex_table(rows)


def print_latex_table(rows: list[dict]) -> None:
    print("\n--- LaTeX snippet (tab:benchmark) ---")
    for row in rows:
        mode = row["mode"]
        tp = row["throughput_rps"] if row["throughput_rps"] is not None else "--"
        p95 = row["latency_p95_ms"] if row["latency_p95_ms"] is not None else "--"
        skip = row["gate_skip_pct"]
        skip_str = f"{skip:.1f}" if skip is not None else "N/A"
        f1 = row["attack_f1"] if row["attack_f1"] is not None else "--"
        print(f"    {mode} & {tp} & {p95} & {skip_str} & {f1} \\\\")


def default_output_path(kind: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paper_out = os.path.join(root, "..", "papers", "soict2026", "results", "benchmarks")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    node = EDGE_NODE_ID
    return os.path.join(paper_out, f"{kind}_{node}_{stamp}.json")


def cmd_local(args: argparse.Namespace) -> None:
    payload = run_local_benchmark(args.samples, args.batch_size)
    payload["deploy_mode"] = args.mode
    out = args.output or default_output_path("local")
    save_json(payload, out)


def cmd_send(args: argparse.Namespace) -> None:
    payload = send_kafka_traffic(args.duration, args.rate, args.csv, args.topic)
    out = args.output or default_output_path("send")
    save_json(payload, out)


def cmd_collect(args: argparse.Namespace) -> None:
    payload = collect_metrics(
        args.window_minutes, args.mode,
        start_epoch=args.start_epoch, end_epoch=args.end_epoch,
        raw_latency_glob=args.raw_latency_glob,
    )
    out = args.output or default_output_path("collect")
    save_json(payload, out)


def cmd_run(args: argparse.Namespace) -> None:
    for rep in range(args.repeats):
        if args.repeats > 1:
            print(f"\n===== Repeat {rep + 1}/{args.repeats} (mode={args.mode}) =====")
        payload = run_end_to_end(
            deploy_mode=args.mode,
            duration_s=args.duration,
            rate=args.rate,
            warmup_s=args.warmup if rep == 0 else max(args.warmup // 3, 5),
            window_minutes=args.window_minutes,
            csv_path=args.csv,
            raw_latency_glob=args.raw_latency_glob,
        )
        payload["repeat_index"] = rep
        if args.repeats > 1 or not args.output:
            out = default_output_path(f"run_{args.mode}_rep{rep}")
        else:
            out = args.output
        save_json(payload, out)
        print_latex_table([row_from_result(payload)])
        if rep < args.repeats - 1:
            time.sleep(args.cooldown)
    if args.repeats < 3:
        print(f"\n[WARN] Only {args.repeats} repeat(s). Use --repeats 5 for "
              "publishable mean±std.")


def cmd_node_power(args: argparse.Namespace) -> None:
    """Per-node energy measurement DURING a distributed run.

    Start this on EACH Jetson before the orchestrator sends load, with
    --duration >= warmup + load duration. It measures the idle baseline first,
    then samples tegrastats for the window and reports raw + idle-subtracted
    (active) energy. In pipeline-split mode the paper figure is the SUM of both
    nodes' active energy divided by the number of classified flows.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from edge.power_monitor import PowerMonitor, measure_idle_power

    print(f"[POWER] Measuring idle baseline for {args.idle_seconds}s "
          "(make sure NO load is being sent)...")
    idle_w = measure_idle_power(seconds=args.idle_seconds)
    if idle_w is None:
        print("[ERR] tegrastats unavailable — is this a Jetson?")
        sys.exit(1)
    print(f"[POWER] Idle baseline: {idle_w:.2f} W. Sampling for {args.duration}s — "
          "start the load now.")

    pm = PowerMonitor(idle_power_w=idle_w).start()
    time.sleep(args.duration)
    stats = pm.stop()
    payload = {
        "benchmark_type": "node_power",
        "node_id": EDGE_NODE_ID,
        "role": EDGE_NODE_ROLE,
        "device": detect_device_name(),
        "idle_seconds": args.idle_seconds,
        "timestamp": utc_now_iso(),
        **stats,
    }
    out = args.output or default_output_path("power")
    save_json(payload, out)


def cmd_merge(args: argparse.Namespace) -> None:
    merge_results(args.input, args.output_csv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed IDS benchmark orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_local = sub.add_parser("local", help="Local PySpark inference micro-benchmark")
    p_local.add_argument("--samples", type=int, default=500)
    p_local.add_argument("--batch-size", type=int, default=10)
    p_local.add_argument("--mode", default=EDGE_NODE_ROLE, help="Deploy mode label")
    p_local.add_argument("--output", default=None)
    p_local.set_defaults(func=cmd_local)

    p_send = sub.add_parser("send", help="Send Kafka traffic for a fixed duration")
    p_send.add_argument("--duration", type=int, default=60)
    p_send.add_argument("--rate", type=int, default=100)
    p_send.add_argument("--csv", default=None)
    p_send.add_argument("--topic", default=KAFKA_TOPIC)
    p_send.add_argument("--output", default=None)
    p_send.set_defaults(func=cmd_send)

    p_collect = sub.add_parser("collect", help="Query InfluxDB/Postgres metrics")
    p_collect.add_argument("--window-minutes", type=int, default=5)
    p_collect.add_argument("--mode", default=os.getenv("BENCHMARK_DEPLOY_MODE", "split"))
    p_collect.add_argument("--start-epoch", type=float, default=None,
                           help="Load window start (unix epoch) to align queries")
    p_collect.add_argument("--end-epoch", type=float, default=None,
                           help="Load window end (unix epoch)")
    p_collect.add_argument("--raw-latency-glob", default=None,
                           help="Glob of RAW_LATENCY_LOG CSVs pulled from the nodes "
                                "(required for publishable run-level p95)")
    p_collect.add_argument("--output", default=None)
    p_collect.set_defaults(func=cmd_collect)

    p_run = sub.add_parser("run", help="Warmup (real load) + send + collect (distributed)")
    p_run.add_argument("--mode", choices=["single", "split", "horizontal", "spark_cluster"], default="split")
    p_run.add_argument("--duration", type=int, default=60)
    p_run.add_argument("--rate", type=int, default=100)
    p_run.add_argument("--warmup", type=int, default=30,
                       help="Seconds of REAL warmup traffic before the measured window")
    p_run.add_argument("--repeats", type=int, default=5,
                       help="Number of measured repetitions (>=3 for mean±std)")
    p_run.add_argument("--cooldown", type=int, default=20,
                       help="Seconds between repetitions")
    p_run.add_argument("--window-minutes", type=int, default=5)
    p_run.add_argument("--csv", default=None)
    p_run.add_argument("--raw-latency-glob", default=None,
                       help="Glob of RAW_LATENCY_LOG CSVs (mounted/synced from nodes)")
    p_run.add_argument("--output", default=None)
    p_run.set_defaults(func=cmd_run)

    p_power = sub.add_parser("node-power",
                             help="Per-node tegrastats energy sampling during a run")
    p_power.add_argument("--duration", type=int, default=90,
                         help="Sampling window (>= warmup + load duration)")
    p_power.add_argument("--idle-seconds", type=float, default=30.0,
                         help="Idle baseline sampling time BEFORE load starts")
    p_power.add_argument("--output", default=None)
    p_power.set_defaults(func=cmd_node_power)

    p_merge = sub.add_parser("merge", help="Merge JSON results into paper CSV")
    p_merge.add_argument("--input", default="../../papers/soict2026/results/benchmarks/*.json")
    p_merge.add_argument("--output-csv", default="../../papers/soict2026/results/benchmarks/summary.csv")
    p_merge.set_defaults(func=cmd_merge)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
