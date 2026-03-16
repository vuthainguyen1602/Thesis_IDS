#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Model Benchmark — Compare DT, GBT, RF on Raspberry Pi.

Runs identical workloads against multiple saved PySpark PipelineModels to
produce side-by-side throughput, latency, and resource-usage comparisons.

Usage (on Raspberry Pi)::

    python scripts/benchmark_all.py
    python scripts/benchmark_all.py --samples 500

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import time
import json
import argparse
import statistics

import psutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_PATH, SHAP_TOP_FEATURES


def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def generate_data(feature_columns, n):
    return [{col: float(np.random.uniform(0, 1000)) for col in feature_columns} for _ in range(n)]


def benchmark_model(spark, model_path, model_name, test_data, feature_columns, batch_size=10):
    """Benchmark a single model."""
    from pyspark.ml import PipelineModel
    from edge.feature_preprocessor import FeaturePreprocessor

    print(f"\n{'=' * 50}")
    print(f"  Benchmarking: {model_name}")
    print(f"{'=' * 50}")

    # Load model
    load_start = time.time()
    preprocessor = FeaturePreprocessor(spark)
    model = PipelineModel.load(model_path)
    model_load_time = time.time() - load_start
    print(f"  Model loaded in {model_load_time:.1f}s")

    n_samples = len(test_data)

    # Warmup
    for i in range(0, min(30, n_samples), batch_size):
        batch = test_data[i:i + batch_size]
        df = preprocessor.preprocess_batch(batch)
        model.transform(df).count()

    # Benchmark
    batch_latencies = []
    cpu_readings = []
    mem_readings = []
    temp_readings = []

    benchmark_start = time.time()

    for i in range(0, n_samples, batch_size):
        batch = test_data[i:i + batch_size]

        cpu_readings.append(psutil.cpu_percent(interval=None))
        mem_readings.append(psutil.virtual_memory().percent)
        temp = get_cpu_temp()
        if temp:
            temp_readings.append(temp)

        t0 = time.perf_counter()
        df = preprocessor.preprocess_batch(batch)
        predictions = model.transform(df)
        predictions.count()
        batch_time = (time.perf_counter() - t0) * 1000
        batch_latencies.append(batch_time)

    total_time = time.time() - benchmark_start
    throughput = n_samples / total_time
    avg_latency = statistics.mean(batch_latencies)
    p50 = statistics.median(batch_latencies)
    p95 = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
    per_sample = avg_latency / batch_size

    result = {
        "model": model_name,
        "throughput_rps": round(throughput, 1),
        "latency_mean_ms": round(avg_latency, 1),
        "latency_per_sample_ms": round(per_sample, 1),
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "cpu_avg": round(statistics.mean(cpu_readings), 1),
        "ram_avg": round(statistics.mean(mem_readings), 1),
        "temp_avg": round(statistics.mean(temp_readings), 1) if temp_readings else None,
        "model_load_time_s": round(model_load_time, 1),
    }

    print(f"  Throughput:  {throughput:.1f} rps")
    print(f"  Latency:     {per_sample:.1f} ms/sample")
    print(f"  CPU:         {result['cpu_avg']}% | RAM: {result['ram_avg']}%")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    print("\n" + "=" * 60)
    print("  MULTI-MODEL RPi BENCHMARK")
    print(f"  Samples: {args.samples} | Batch: {args.batch_size}")
    print("=" * 60)

    spark = (
        SparkSession.builder
        .appName("IDS_Benchmark_All")
        .master("local[2]")
        .config("spark.executor.memory", "1g")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Load feature columns
    feature_columns = SHAP_TOP_FEATURES
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            feature_columns = json.load(f)

    test_data = generate_data(feature_columns, args.samples)

    # Models to benchmark
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
    models = [
        ("Decision Tree", os.path.join(model_dir, "ids_pipeline_decision_tree")),
        ("GBT", os.path.join(model_dir, "ids_pipeline_gbt")),
        ("Random Forest", os.path.join(model_dir, "ids_pipeline_random_forest")),
    ]

    all_results = []
    for name, path in models:
        if os.path.exists(path):
            result = benchmark_model(spark, path, name, test_data, feature_columns, args.batch_size)
            all_results.append(result)
        else:
            print(f"\n  [WARN] Model not found: {path}")

    # Summary table
    print("\n" + "=" * 60)
    print("  COMPARISON RESULTS")
    print("=" * 60)
    print(f"\n  {'Model':<18} {'Throughput':>10} {'Latency':>10} {'CPU':>6} {'RAM':>6}")
    print(f"  {'-' * 52}")
    for r in all_results:
        print(f"  {r['model']:<18} {r['throughput_rps']:>8.1f}/s {r['latency_per_sample_ms']:>8.1f}ms {r['cpu_avg']:>5.1f}% {r['ram_avg']:>5.1f}%")

    # Save
    results_path = os.path.join(model_dir, "benchmark_comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    spark.stop()
    print("\n[OK] All benchmarks complete.")


if __name__ == "__main__":
    main()
