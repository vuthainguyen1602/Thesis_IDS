#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot edge deployment-mode comparison for SOICT'2026.

Reads the merged edge benchmark CSV produced by run_benchmarks.sh
(`results/benchmarks/summary.csv`) and draws a grouped bar chart of
throughput (req/s) and p95 latency (ms) per deployment mode:
  single / split (Mode A) / horizontal (Mode B) / spark_cluster (Mode C).

Rows with empty throughput (unfilled template) are skipped, so this can be run
at any time; it only plots modes that have real numbers.

Output: results/benchmarks/edge_modes.png  (in the SOICT \\graphicspath)

Usage:
  python papers/soict2026/plot_edge_modes.py
  python papers/soict2026/plot_edge_modes.py --csv path/to/summary.csv
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "results", "benchmarks", "summary.csv")

MODE_LABELS = {
    "single": "Single node",
    "split": "A: Pipeline split",
    "horizontal": "B: Horizontal",
    "spark_cluster": "C: Spark cluster",
}


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[ERROR] Benchmark CSV not found: {args.csv}\n"
                 "        Run ./run_benchmarks.sh (local|run|merge) first.")

    df = pd.read_csv(args.csv)
    for col in ("throughput_rps", "latency_p95_ms"):
        if col not in df.columns:
            sys.exit(f"[ERROR] Column '{col}' missing in {args.csv}")
        df[col] = _to_num(df[col])

    # keep only rows with a real throughput measurement
    df = df[df["throughput_rps"].notna()].copy()
    if df.empty:
        sys.exit("[WARN] No filled benchmark rows yet (throughput all empty). "
                 "Fill summary.csv after running the benchmarks.")

    # one row per mode (first filled occurrence), in canonical order
    order = [m for m in MODE_LABELS if m in set(df["mode"])]
    df = df.drop_duplicates("mode").set_index("mode").loc[order]
    labels = [MODE_LABELS[m] for m in order]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(order))
    fig, ax1 = plt.subplots(figsize=(8, 5))

    b1 = ax1.bar(x - 0.2, df["throughput_rps"].values, 0.4,
                 color="#1f77b4", label="Throughput (req/s)")
    ax1.set_ylabel("Throughput (req/s)", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.2, df["latency_p95_ms"].values, 0.4,
                 color="#d62728", label="p95 latency (ms)")
    ax2.set_ylabel("p95 latency (ms)", color="#d62728")

    ax1.bar_label(b1, fmt="%.0f", fontsize=8, padding=2)
    ax2.bar_label(b2, fmt="%.0f", fontsize=8, padding=2)
    ax1.set_title("Edge deployment modes: throughput vs. p95 latency")
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.csv), "edge_modes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    main()
