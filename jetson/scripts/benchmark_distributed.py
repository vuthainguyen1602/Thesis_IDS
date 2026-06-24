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


def query_influx_metrics(window_minutes: int) -> dict[str, dict]:
    """Aggregate throughput and latency per host from InfluxDB."""
    try:
        from influxdb_client import InfluxDBClient
    except ImportError:
        print("[WARN] influxdb-client not installed; skipping InfluxDB query")
        return {}

    flux = f"""
from(bucket: "{INFLUXDB_BUCKET}")
  |> range(start: -{window_minutes}m)
  |> filter(fn: (r) => r._measurement == "prediction_metrics")
  |> filter(fn: (r) => r._field == "throughput_rps" or r._field == "avg_latency_ms")
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


def query_postgres_gate_skip(window_minutes: int) -> dict:
    """Estimate gate skip ratio from stored predictions."""
    try:
        import psycopg2
    except ImportError:
        print("[WARN] psycopg2 not installed; skipping Postgres query")
        return {}

    since = time.time() - window_minutes * 60
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
                WHERE timestamp >= %s
                """,
                (since,),
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


def query_system_metrics(window_minutes: int) -> dict[str, dict]:
    try:
        from influxdb_client import InfluxDBClient
    except ImportError:
        return {}

    flux = f"""
from(bucket: "{INFLUXDB_BUCKET}")
  |> range(start: -{window_minutes}m)
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


def collect_metrics(window_minutes: int, deploy_mode: str) -> dict:
    influx_pred = query_influx_metrics(window_minutes)
    influx_sys = query_system_metrics(window_minutes)
    gate = query_postgres_gate_skip(window_minutes)

    hosts = sorted(set(influx_pred) | set(influx_sys))
    nodes = []
    throughputs = []
    latencies = []

    for host in hosts:
        pred = influx_pred.get(host, {})
        sys_m = influx_sys.get(host, {})
        tp = pred.get("throughput_rps", 0.0)
        lat = pred.get("avg_latency_ms", 0.0)
        if tp > 0:
            throughputs.append(tp)
        if lat > 0:
            latencies.append(lat)
        nodes.append({
            "node_id": host,
            "throughput_rps": tp,
            "avg_latency_ms": lat,
            "cpu_percent": sys_m.get("cpu_percent"),
            "memory_percent": sys_m.get("memory_percent"),
            "cpu_temp_celsius": sys_m.get("cpu_temp_celsius"),
        })

    aggregate = {
        "throughput_rps_sum": round(sum(throughputs), 2) if throughputs else None,
        "throughput_rps_mean": round(statistics.mean(throughputs), 2) if throughputs else None,
        "latency_avg_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_p95_ms": round(percentile(latencies, 0.95), 2) if latencies else None,
    }

    return {
        "benchmark_type": "collect",
        "deploy_mode": deploy_mode,
        "window_minutes": window_minutes,
        "timestamp": utc_now_iso(),
        "gate_skip": gate,
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
) -> dict:
    print(f"\n[WAIT] Warmup {warmup_s}s — ensure edge pipelines are running...")
    time.sleep(warmup_s)

    send_info = send_kafka_traffic(duration_s, rate, csv_path, KAFKA_TOPIC)
    print(f"\n[WAIT] Draining metrics for {window_minutes} min window...")
    time.sleep(min(30, window_minutes * 10))

    collect_info = collect_metrics(window_minutes, deploy_mode)
    return {
        "benchmark_type": "run",
        "deploy_mode": deploy_mode,
        "send": send_info,
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
        "throughput_rps": agg.get("throughput_rps_sum") or data.get("throughput_rps"),
        "latency_p95_ms": agg.get("latency_p95_ms") or data.get("latency_batch_p95_ms"),
        "latency_avg_ms": agg.get("latency_avg_ms") or data.get("latency_per_sample_ms"),
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
        "gate_skip_pct", "attack_f1", "cpu_percent", "memory_percent", "device", "timestamp",
    ]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Merged {len(rows)} runs -> {output_csv}")
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
    payload = collect_metrics(args.window_minutes, args.mode)
    out = args.output or default_output_path("collect")
    save_json(payload, out)


def cmd_run(args: argparse.Namespace) -> None:
    payload = run_end_to_end(
        deploy_mode=args.mode,
        duration_s=args.duration,
        rate=args.rate,
        warmup_s=args.warmup,
        window_minutes=args.window_minutes,
        csv_path=args.csv,
    )
    out = args.output or default_output_path(f"run_{args.mode}")
    save_json(payload, out)
    print_latex_table([row_from_result(payload)])


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
    p_collect.add_argument("--output", default=None)
    p_collect.set_defaults(func=cmd_collect)

    p_run = sub.add_parser("run", help="Warmup + send + collect (distributed)")
    p_run.add_argument("--mode", choices=["single", "split", "horizontal", "spark_cluster"], default="split")
    p_run.add_argument("--duration", type=int, default=60)
    p_run.add_argument("--rate", type=int, default=100)
    p_run.add_argument("--warmup", type=int, default=15)
    p_run.add_argument("--window-minutes", type=int, default=5)
    p_run.add_argument("--csv", default=None)
    p_run.add_argument("--output", default=None)
    p_run.set_defaults(func=cmd_run)

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
