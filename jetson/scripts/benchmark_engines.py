#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-node inference-engine baseline: scikit-learn (and optional ONNX Runtime)
vs the PySpark classifier benchmarked by ``benchmark.py``.

Why this exists
---------------
The thesis deploys a Spark ``PipelineModel`` on the edge for **consistency with
the distributed training phase** — the same stack trains (on the cluster) and
serves. That unification has a cost: a JVM + Spark scheduler on an 8 GB ARM
board is heavy for a small (~3 MB) tree model. This script *quantifies* that
cost honestly by running the SAME algorithm (RandomForest, matched hyper-
parameters) on the SAME SHAP Top-30 features through lighter engines:

  * scikit-learn (pure-CPU, no JVM)
  * ONNX Runtime (if ``skl2onnx`` + ``onnxruntime`` are installed)

It reports the same metrics as ``benchmark.py`` (throughput, p50/p95/p99 latency
from raw samples, CPU/RAM, idle-subtracted energy) so the two JSON outputs can
be placed side by side. The point is NOT that one "wins": it is the measured
trade-off — Spark unifies train+serve; sklearn/ONNX cut inference cost and
motivate an ONNX/TensorRT export path for deployment as future work.

Accuracy is comparable by construction (same algorithm + features); this
micro-benchmark targets inference cost. With ``--csv`` it trains on real flows;
otherwise it uses synthetic noise (latency/throughput/energy are representative,
but the attack rate is meaningless — same caveat as ``benchmark.py``).

Usage:
    python scripts/benchmark_engines.py --samples 5000 --batch-size 10
    python scripts/benchmark_engines.py --csv /path/train.csv --engines sklearn,onnx
"""
import os
import sys
import json
import time
import argparse
import statistics

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_PATH, SHAP_TOP_FEATURES  # noqa: E402
from edge.power_monitor import PowerMonitor, measure_idle_power  # noqa: E402

try:
    import psutil
except Exception:  # psutil is optional for the resource readings
    psutil = None


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 3)
    return round(sorted_vals[int(round(q * (len(sorted_vals) - 1)))], 3)


def _feature_columns():
    cols = SHAP_TOP_FEATURES
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            cols = json.load(f)
    return cols


def _load_xy(csv_path, feature_cols, n_samples, seed=42):
    """Return (X, y) as float32 ndarrays. Real CSV if given, else synthetic."""
    rng = np.random.default_rng(seed)
    if csv_path and os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        present = [c for c in feature_cols if c in df.columns]
        X = df[present].apply(lambda s: s.astype("float32"), axis=0)
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32)
        if "label" in df.columns:
            y = (df["label"].astype(str).str.upper() != "BENIGN").astype(np.int64).to_numpy()
        else:
            y = rng.integers(0, 2, size=len(X))
        return X[:n_samples], y[:n_samples], present
    # Synthetic: uniform noise; labels random (latency-only, attack rate meaningless)
    X = rng.uniform(0, 1000, size=(n_samples, len(feature_cols))).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples)
    return X, y, feature_cols


def _resource_snapshot():
    if psutil is None:
        return None, None
    return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent


def _bench_predict(predict_fn, X, batch_size, label):
    """Time batched inference; return raw-sample latency stats + resources."""
    n = len(X)
    # Warmup (JVM-free here, but warms caches / JIT in numpy/ort)
    predict_fn(X[: min(50, n)])

    cpu_readings, mem_readings = [], []
    batch_latencies = []
    idle_w = measure_idle_power(seconds=3.0)
    power = PowerMonitor(idle_power_w=idle_w).start()

    t0 = time.time()
    total = 0
    for i in range(0, n, batch_size):
        batch = X[i: i + batch_size]
        c, m = _resource_snapshot()
        if c is not None:
            cpu_readings.append(c); mem_readings.append(m)
        bt = time.perf_counter()
        predict_fn(batch)
        batch_latencies.append((time.perf_counter() - bt) * 1000.0)
        total += len(batch)
    wall = time.time() - t0
    pstats = power.stop()

    bl = sorted(batch_latencies)
    energy_active_mj = (round(pstats["energy_active_j"] * 1000.0 / total, 3)
                        if pstats.get("energy_active_j") is not None and total else None)
    energy_raw_mj = (round(pstats["energy_j"] * 1000.0 / total, 3)
                     if pstats.get("energy_j") is not None and total else None)
    return {
        "engine": label,
        "samples": total,
        "batch_size": batch_size,
        "throughput_rps": round(total / wall, 1) if wall > 0 else None,
        "latency_batch_mean_ms": round(statistics.mean(batch_latencies), 3),
        "latency_batch_p50_ms": _percentile(bl, 0.50),
        "latency_batch_p95_ms": _percentile(bl, 0.95),
        "latency_batch_p99_ms": _percentile(bl, 0.99),
        "avg_cpu_percent": round(statistics.mean(cpu_readings), 1) if cpu_readings else None,
        "avg_memory_percent": round(statistics.mean(mem_readings), 1) if mem_readings else None,
        "idle_power_w": pstats.get("idle_power_w"),
        "active_power_w": pstats.get("active_power_w"),
        "energy_per_inference_mj_raw": energy_raw_mj,
        "energy_per_inference_mj_active": energy_active_mj,
        "power_available": pstats.get("power_available", False),
    }


def run(n_samples, batch_size, csv_path, engines):
    feature_cols = _feature_columns()
    X, y, used_cols = _load_xy(csv_path, feature_cols, n_samples)
    print(f"[DATA] {len(X)} samples x {len(used_cols)} features "
          f"({'real CSV' if csv_path else 'synthetic noise'})")

    from sklearn.ensemble import RandomForestClassifier
    # Match the Spark RandomForest hyper-parameters for a fair comparison.
    print("[FIT] sklearn RandomForest(n_estimators=200, max_depth=15)...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, max_features="sqrt",
        n_jobs=-1, random_state=42,
    ).fit(X, y)
    model_size_mb = None
    try:
        import pickle
        model_size_mb = round(len(pickle.dumps(clf)) / (1024 * 1024), 3)
    except Exception:
        pass

    results = []
    if "sklearn" in engines:
        results.append(_bench_predict(lambda b: clf.predict(b), X, batch_size, "sklearn"))

    if "onnx" in engines:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnxruntime as ort
            onx = convert_sklearn(
                clf, initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
                options={id(clf): {"zipmap": False}},
            )
            sess = ort.InferenceSession(onx.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            results.append(_bench_predict(
                lambda b: sess.run([out_name], {in_name: b.astype(np.float32)}),
                X, batch_size, "onnxruntime"))
        except ImportError:
            print("[SKIP] ONNX engine: install skl2onnx + onnxruntime to enable.")
        except Exception as e:
            print(f"[SKIP] ONNX engine failed: {e}")

    print("\n" + "=" * 78)
    print("  INFERENCE-ENGINE BASELINE (compare against benchmark.py for Spark)")
    print("=" * 78)
    hdr = f"  {'Engine':<12}{'Tput(rps)':>11}{'p50(ms)':>9}{'p95(ms)':>9}{'RAM%':>7}{'mJ/inf(act)':>13}"
    print(hdr)
    for r in results:
        print(f"  {r['engine']:<12}{str(r['throughput_rps']):>11}"
              f"{str(r['latency_batch_p50_ms']):>9}{str(r['latency_batch_p95_ms']):>9}"
              f"{str(r['avg_memory_percent']):>7}{str(r['energy_per_inference_mj_active']):>13}")
    print(f"\n  sklearn model size (pickle): {model_size_mb} MB")
    print("  [NOTE] Spark unifies train+serve; these engines cut inference cost but")
    print("         would need an export step (PipelineModel -> ONNX) for deployment.")
    if not csv_path:
        print("  [NOTE] Synthetic data: throughput/latency/energy representative; accuracy N/A.")

    out = {
        "device": "single-node baseline",
        "data": "real_csv" if csv_path else "synthetic",
        "features": len(used_cols),
        "sklearn_model_size_mb": model_size_mb,
        "engines": results,
    }
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark_engines_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inference-engine baseline (sklearn/ONNX vs Spark)")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--csv", type=str, default=None, help="Real flows CSV (optional)")
    ap.add_argument("--engines", type=str, default="sklearn,onnx",
                    help="Comma list: sklearn,onnx")
    a = ap.parse_args()
    run(a.samples, a.batch_size, a.csv, set(e.strip() for e in a.engines.split(",")))
