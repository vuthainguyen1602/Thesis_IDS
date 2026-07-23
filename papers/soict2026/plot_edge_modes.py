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
# summary_paper.csv holds the run-level worst-node p95 from raw per-flow logs
# (the same aggregation reported in the paper's Table 4); summary.csv holds
# per-node monitor values and does NOT match the table.
DEFAULT_CSV = os.path.join(HERE, "results", "benchmarks", "summary_paper.csv")

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

    # aggregate repeated runs: mean per mode, in canonical order
    order = [m for m in MODE_LABELS if m in set(df["mode"])]
    df = (df.groupby("mode")[["throughput_rps", "latency_p95_ms"]]
            .mean().loc[order])
    labels = [MODE_LABELS[m] for m in order]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Two side-by-side panels (one axis each) instead of a dual-axis chart.
    x = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)

    b1 = ax1.bar(x, df["throughput_rps"].values, 0.55, color="#0072B2")
    ax1.set_ylabel("Throughput (flows/s)")
    ax1.set_title("Verdict throughput", fontsize=10)

    b2 = ax2.bar(x, df["latency_p95_ms"].values, 0.55, color="#D55E00")
    ax2.set_ylabel("p95 latency (ms)")
    ax2.set_title("p95 latency", fontsize=10)

    for ax, bars in ((ax1, b1), (ax2, b2)):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.csv), "edge_modes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    main()
