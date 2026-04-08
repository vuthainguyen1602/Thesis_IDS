#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
import statistics

import psutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, FEATURES_PATH, SHAP_TOP_FEATURES


def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def generate_synthetic_data(feature_columns, n_samples):
    data = []
    for _ in range(n_samples):
        row = {}
        for col in feature_columns:
            row[col] = float(np.random.uniform(0, 1000))
        data.append(row)
    return data


def run_benchmark(n_samples=1000, batch_size=10):
    from pyspark.sql import SparkSession
    from edge.feature_preprocessor import FeaturePreprocessor
    from edge.prediction_engine import PredictionEngine

    print("\n" + "=" * 60)
    print("  IDS EDGE BENCHMARK")
    print("=" * 60)
    print(f"  Samples:    {n_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model:      {MODEL_PATH}")
    print("=" * 60)

    print("\n[1/4] Initializing PySpark...")
    spark_start = time.time()
    spark = (
        SparkSession.builder
        .appName("IDS_Benchmark")
        .master("local[2]")
        .config("spark.executor.memory", "1g")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    spark_init_time = time.time() - spark_start
    print(f"  Spark init: {spark_init_time:.1f}s")

    print("\n[2/4] Loading model...")
    model_start = time.time()
    preprocessor = FeaturePreprocessor(spark)
    engine = PredictionEngine(spark)
    model_load_time = time.time() - model_start
    print(f"  Model load: {model_load_time:.1f}s")

    print("\n[3/4] Generating test data...")
    feature_columns = SHAP_TOP_FEATURES
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            feature_columns = json.load(f)

    test_data = generate_synthetic_data(feature_columns, n_samples)
    print(f"  Generated {n_samples} samples x {len(feature_columns)} features")

    print(f"\n[4/4] Running inference benchmark...")

    batch_latencies = []
    per_sample_latencies = []
    cpu_readings = []
    mem_readings = []
    temp_readings = []
    total_predictions = 0
    total_attacks = 0

    print("  Warmup...", end="", flush=True)
    for i in range(0, min(30, n_samples), batch_size):
        batch = test_data[i:i + batch_size]
        spark_df = preprocessor.preprocess_batch(batch)
        engine.predict(spark_df)
    print(" done")

    engine.total_predictions = 0
    engine.total_attacks = 0
    engine.total_inference_time = 0

    benchmark_start = time.time()
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        batch = test_data[i:i + batch_size]
        actual_batch_size = len(batch)

        cpu_readings.append(psutil.cpu_percent(interval=None))
        mem_readings.append(psutil.virtual_memory().percent)
        temp = get_cpu_temp()
        if temp:
            temp_readings.append(temp)

        batch_start = time.perf_counter()
        spark_df = preprocessor.preprocess_batch(batch)
        predictions, stats = engine.predict(spark_df)
        batch_time = (time.perf_counter() - batch_start) * 1000

        batch_latencies.append(batch_time)
        per_sample_latencies.append(batch_time / actual_batch_size)
        total_predictions += stats["batch_size"]
        total_attacks += stats["attacks_found"]
        n_batches += 1

        if (i + batch_size) % (n_samples // 10 if n_samples >= 10 else 1) == 0:
            pct = min(100, (i + batch_size) / n_samples * 100)
            print(f"  Progress: {pct:.0f}% ({i + batch_size}/{n_samples})")

    total_time = time.time() - benchmark_start

    throughput = total_predictions / total_time
    avg_batch_latency = statistics.mean(batch_latencies)
    p50_batch = statistics.median(batch_latencies)
    p95_batch = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
    p99_batch = sorted(batch_latencies)[int(len(batch_latencies) * 0.99)]
    avg_per_sample = statistics.mean(per_sample_latencies)
    avg_cpu = statistics.mean(cpu_readings) if cpu_readings else 0
    avg_mem = statistics.mean(mem_readings) if mem_readings else 0
    avg_temp = statistics.mean(temp_readings) if temp_readings else 0

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n  --- Inference Performance ---")
    print(f"  Total samples:        {total_predictions:,}")
    print(f"  Total time:           {total_time:.2f}s")
    print(f"  Throughput:           {throughput:.1f} samples/sec")
    print(f"  Batch size:           {batch_size}")
    print(f"  Num batches:          {n_batches}")

    print(f"\n  --- Latency (per batch of {batch_size}) ---")
    print(f"  Mean:                 {avg_batch_latency:.1f} ms")
    print(f"  Median (P50):         {p50_batch:.1f} ms")
    print(f"  P95:                  {p95_batch:.1f} ms")
    print(f"  P99:                  {p99_batch:.1f} ms")

    print(f"\n  --- Latency (per sample) ---")
    print(f"  Mean:                 {avg_per_sample:.1f} ms")

    print(f"\n  --- System Resources ---")
    print(f"  Avg CPU:              {avg_cpu:.1f}%")
    print(f"  Avg Memory:           {avg_mem:.1f}%")
    if avg_temp:
        print(f"  Avg Temperature:      {avg_temp:.1f} C")

    print(f"\n  --- Model Info ---")
    print(f"  Spark init time:      {spark_init_time:.1f}s")
    print(f"  Model load time:      {model_load_time:.1f}s")
    print(f"  Attack rate:          {total_attacks/total_predictions*100:.1f}%")

    results = {
        "device": "Raspberry Pi 4B (4GB)",
        "model": "PySpark RandomForest (200 trees, depth 15)",
        "features": f"SHAP Top-{len(feature_columns)}",
        "total_samples": total_predictions,
        "total_time_s": round(total_time, 2),
        "throughput_rps": round(throughput, 1),
        "batch_size": batch_size,
        "latency_batch_mean_ms": round(avg_batch_latency, 1),
        "latency_batch_p50_ms": round(p50_batch, 1),
        "latency_batch_p95_ms": round(p95_batch, 1),
        "latency_batch_p99_ms": round(p99_batch, 1),
        "latency_per_sample_ms": round(avg_per_sample, 1),
        "avg_cpu_percent": round(avg_cpu, 1),
        "avg_memory_percent": round(avg_mem, 1),
        "avg_temp_celsius": round(avg_temp, 1) if avg_temp else None,
        "spark_init_time_s": round(spark_init_time, 1),
        "model_load_time_s": round(model_load_time, 1),
    }

    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print(f"\n  --- LaTeX Table (for Chapter 4) ---")
    print(r"  \begin{table}[h]")
    print(r"  \centering")
    print(r"  \caption{IDS System Performance on Raspberry Pi 4B}")
    print(r"  \begin{tabular}{lr}")
    print(r"  \toprule")
    print(r"  \textbf{Metric} & \textbf{Value} \\")
    print(r"  \midrule")
    print(f"  Throughput & {throughput:.1f} samples/s \\\\")
    print(f"  Latency (mean) & {avg_per_sample:.1f} ms \\\\")
    print(f"  Latency P95 & {p95_batch:.1f} ms \\\\")
    print(f"  CPU utilization & {avg_cpu:.1f}\\% \\\\")
    print(f"  RAM utilization & {avg_mem:.1f}\\% \\\\")
    if avg_temp:
        print(f"  CPU temperature & {avg_temp:.1f}°C \\\\")
    print(f"  Spark init time & {spark_init_time:.1f}s \\\\")
    print(f"  Model size & 2.9 MB \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \end{table}")

    spark.stop()
    print("\n[OK] Benchmark complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS Edge Benchmark")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    args = parser.parse_args()

    run_benchmark(n_samples=args.samples, batch_size=args.batch_size)
