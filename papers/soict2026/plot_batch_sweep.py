#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch-size sweep of the deployed forest: PySpark vs ONNX Runtime (Jetson #2).

Reads results/batch_sweep_20261011/ (jetson/scripts/sweep_batch_size.sh).
Spark rows come from batch_<n>.json. ONNX rows come from
batch_<n>_onnx_energy.json, whose tiled runs last long enough for tegrastats
to return energy; the short ONNX runs in batch_<n>.json are only used to check
that Spark and ONNX agree.

    python papers/soict2026/plot_batch_sweep.py
    python papers/soict2026/plot_batch_sweep.py --vi thesis/img/batch_sweep_vi.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "batch_sweep_20261011")
COLORS = {"spark": "#2a78d6", "onnx": "#eb6834"}
LABELS = {"spark": "PySpark (deployed)", "onnx": "ONNX Runtime"}
TITLES = {"en": ("Time per batch, p50 (s)", "Throughput (flows/s)", "Active energy per flow (mJ)"),
          "vi": ("Thời gian mỗi batch, p50 (s)", "Thông lượng (luồng/s)", "Năng lượng mỗi luồng (mJ)")}
XLABEL = {"en": "Batch size (flows)", "vi": "Kích thước batch (luồng)"}


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "batch_*.json"))):
        doc = json.load(open(f))
        agree = doc.get("agreement_pct", {}).get("spark_vs_onnx")
        if agree is not None and agree != 100.0:
            raise SystemExit(f"{f}: Spark and ONNX disagree ({agree}%)")
        energy_run = f.endswith("_onnx_energy.json")
        for r in doc["engines"]:
            if r["engine"] == "spark" or (r["engine"] == "onnx" and energy_run):
                rows.append(r)
    df = pd.DataFrame(rows).sort_values(["engine", "batch_size"])
    df["batch_p50_s"] = df["latency_batch_p50_ms"] / 1000
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vi", metavar="OUT", help="Vietnamese labels, written to OUT")
    args = ap.parse_args()
    lang = "vi" if args.vi else "en"
    df = load()
    table = df.pivot(index="batch_size", columns="engine",
                     values=["batch_p50_s", "throughput_rps", "energy_active_per_inference_mj"])
    print(table.round(3))
    table.to_csv(os.path.join(D, "batch_sweep_table.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    cols = ("batch_p50_s", "throughput_rps", "energy_active_per_inference_mj")
    for ax, col, title in zip(axes, cols, TITLES[lang]):
        for eng in ("spark", "onnx"):
            g = df[df["engine"] == eng]
            ax.plot(g["batch_size"], g[col], marker="o", markersize=5, linewidth=2,
                    color=COLORS[eng], label=LABELS[eng])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(XLABEL[lang], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", color="#e5e5e5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    # The point of the figure: Spark's per-call cost barely moves with batch size.
    sp = df[df["engine"] == "spark"].set_index("batch_size")
    for bs in (10, 10000):
        if bs in sp.index:
            v = sp.loc[bs, "batch_p50_s"]
            txt = f"{v:.1f} s"
            axes[0].annotate(txt.replace(".", ",") if args.vi else txt, (bs, v),
                             textcoords="offset points", xytext=(4 if bs == 10 else 0, 7),
                             ha="left" if bs == 10 else "center", fontsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out = args.vi or os.path.join(D, "batch_sweep.png")
    fig.savefig(out, dpi=200)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
