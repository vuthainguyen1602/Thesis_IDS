#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inference cost of ONE forest through the edge engines (spark, numpy, onnx).

benchmark_engines.py trains its own scikit-learn forest on the replay CSV, so
its trees are not the deployed ones. This script instead serves the deployed
artifacts: model/ids_pipeline_model (Spark) and the numpy/ONNX exports of that
same forest (scripts/export_edge_artifacts.sh), through the edge's own
InferenceEngine classes and FeatureMatrixBuilder, on replayed CICIDS2017 flows.
It also checks that the engines agree prediction by prediction.

    python scripts/benchmark_same_forest.py --samples 1000 --batch-size 10
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

JETSON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JETSON_DIR)

from edge.feature_matrix import FeatureMatrixBuilder, dtype_for_engine  # noqa: E402
from edge.inference_engine import create_inference_engine  # noqa: E402
from edge.power_monitor import PowerMonitor, measure_idle_power  # noqa: E402


def load_rows(path, n):
    with open(path) as fh:
        rows = []
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= n:
                break
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def bench(engine_name, rows, batch_size, spark=None):
    engine = create_inference_engine(spark=spark, engine=engine_name)
    builder = FeatureMatrixBuilder(dtype=dtype_for_engine(engine_name))
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    engine.predict_batch(builder.build(batches[0]))  # warm-up (JIT, first-call cost)

    idle_w = measure_idle_power(seconds=5.0)
    power = PowerMonitor(idle_power_w=idle_w).start()
    preds, lat = [], []
    t0 = time.perf_counter()
    for b in batches:
        s = time.perf_counter()
        out = engine.predict_batch(builder.build(b))
        lat.append((time.perf_counter() - s) * 1000.0)
        preds.append(out.predictions)
    total = time.perf_counter() - t0
    p = power.stop()
    n = len(rows)
    result = {
        "engine": engine_name,
        "samples": n,
        "batch_size": batch_size,
        "throughput_rps": round(n / total, 1),
        "latency_batch_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "latency_batch_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "idle_power_w": p.get("idle_power_w"),
        "avg_power_w": p.get("avg_power_w"),
        "active_power_w": p.get("active_power_w"),
        "energy_active_per_inference_mj": (round(p["energy_active_j"] * 1000.0 / n, 3)
                                           if p.get("energy_active_j") is not None else None),
    }
    return result, np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--csv", default=os.path.join(JETSON_DIR, "sender", "replay_cicids2017.csv"))
    ap.add_argument("--engines", default="spark,numpy,onnx")
    ap.add_argument("--repeat", type=int, default=1,
                    help="tile the rows this many times, so a fast engine runs long enough for tegrastats")
    ap.add_argument("--out", default=os.path.join(JETSON_DIR, "benchmark_same_forest.json"))
    args = ap.parse_args()

    rows = load_rows(args.csv, args.samples) * args.repeat
    results, preds = [], {}
    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        spark = None
        if name == "spark":
            from edge.role_pipelines import create_spark_session
            spark = create_spark_session()
        r, p = bench(name, rows, args.batch_size, spark=spark)
        results.append(r)
        preds[name] = p
        print(r)
        if spark is not None:
            spark.stop()

    names = list(preds)
    agreement = {f"{a}_vs_{b}": round(100.0 * float(np.mean(preds[a] == preds[b])), 3)
                 for i, a in enumerate(names) for b in names[i + 1:]}
    print("prediction agreement (%):", agreement)
    with open(args.out, "w") as fh:
        json.dump({"engines": results, "agreement_pct": agreement}, fh, indent=2)
    print(f"[OK] -> {args.out}")


if __name__ == "__main__":
    main()
