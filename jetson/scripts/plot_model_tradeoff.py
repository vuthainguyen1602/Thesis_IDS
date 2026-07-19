#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Radar (spider) chart of the edge model trade-off for the thesis.

Reads jetson/model/benchmark_comparison.json (produced by the on-device model
benchmark) and draws one polygon per model over five criteria. Each axis is
min-max normalized so that OUTER = BETTER:
  throughput (higher better), latency p95 (lower better), RAM (lower better),
  model size (lower better, from models_info.json), F1 (higher better,
  from models_info.json).

Output: thesis/img/model_tradeoff_radar.png

Usage:  python jetson/scripts/plot_model_tradeoff.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
BENCH = os.path.join(ROOT, "jetson", "model", "benchmark_comparison.json")
INFO = os.path.join(ROOT, "jetson", "model", "models_info.json")
OUT = os.path.join(ROOT, "thesis", "img", "model_tradeoff_radar.png")

AXES = [
    ("Thông lượng", "throughput_rps", True),
    ("Độ trễ p95", "latency_p95_ms", False),
    ("RAM", "ram_avg", False),
    ("Kích thước mô hình", "model_size_mb", False),
    ("F1", "f1", True),
]

COLORS = {"Decision Tree": "#2ca02c", "GBT": "#ff7f0e", "Random Forest": "#1f77b4"}


def main():
    bench = {b["model"]: b for b in json.load(open(BENCH))}
    info = {m["name"]: m for m in json.load(open(INFO))}
    models = [m for m in ("Decision Tree", "GBT", "Random Forest") if m in bench]

    # merge size + f1 from models_info into the benchmark rows
    for m in models:
        bench[m]["model_size_mb"] = info[m]["model_size_mb"]
        bench[m]["f1"] = info[m]["f1"]

    # min-max normalize each axis to [0.15, 1.0] (avoid collapsing to center)
    scores = {m: [] for m in models}
    for _, key, higher_better in AXES:
        vals = np.array([bench[m][key] for m in models], dtype=float)
        lo, hi = vals.min(), vals.max()
        norm = (vals - lo) / (hi - lo) if hi > lo else np.ones_like(vals)
        if not higher_better:
            norm = 1.0 - norm
        norm = 0.15 + 0.85 * norm
        for m, v in zip(models, norm):
            scores[m].append(v)

    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6.4), subplot_kw=dict(polar=True))
    for m in models:
        vals = scores[m] + scores[m][:1]
        ax.plot(angles, vals, linewidth=2, label=m, color=COLORS.get(m))
        ax.fill(angles, vals, alpha=0.12, color=COLORS.get(m))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a[0] for a in AXES], fontsize=11)
    ax.tick_params(axis="x", pad=18)   # push axis labels clear of the circle
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.set_title("Đánh đổi giữa các mô hình trên Jetson Orin Nano Super\n"
                 "(chuẩn hoá min–max, càng ra ngoài càng tốt)", fontsize=12, pad=22)
    ax.legend(loc="lower right", bbox_to_anchor=(1.18, -0.08), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {OUT}")


if __name__ == "__main__":
    main()
