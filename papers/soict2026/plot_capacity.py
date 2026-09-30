#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sustained capacity and energy per verdict per deployment mode (SOICT).

Reads results/remeasure_20260930/capacity_sweep_part{1,2}.csv (analyze_runs.py
rows) and energy_at_sustained_rate.csv. A rate counts as sustained when every
repeat drained within DRAIN_OK seconds; the bar is the mean sustained verdict
rate over the repeats at the highest such rate.

    python papers/soict2026/plot_capacity.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "remeasure_20260930")
DRAIN_OK = 10.0
ORDER = ["single", "single_gate", "horizontal", "spark_cluster", "split"]
LABELS = {"single": "Single node", "single_gate": "Single node\n+ gate",
          "horizontal": "B: Horizontal", "spark_cluster": "C: Spark\ncluster",
          "split": "A: Pipeline\nsplit"}


def capacity_table() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(os.path.join(D, f)) for f in
                    ("capacity_sweep_part1.csv", "capacity_sweep_part2.csv")], ignore_index=True)
    rows = []
    for mode, g in df.groupby("mode"):
        ok = g.groupby("rate").filter(lambda x: (x["drain_s"] <= DRAIN_OK).all() and (x["completed_pct"] >= 99).all())
        if ok.empty:
            continue
        top = ok[ok["rate"] == ok["rate"].max()]
        rows.append({"mode": mode, "rate": top["rate"].iloc[0],
                     "sustained_rps": top["sustained_rps"].mean(),
                     "e2e_p50_s": top["e2e_p50_ms"].mean() / 1000,
                     "e2e_p95_s": top["e2e_p95_ms"].mean() / 1000})
    return pd.DataFrame(rows).set_index("mode")


def main():
    cap = capacity_table()
    en = pd.read_csv(os.path.join(D, "energy_at_sustained_rate.csv")).set_index("mode")
    modes = [m for m in ORDER if m in cap.index]
    print(cap.loc[modes].round(2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    x = range(len(modes))
    colors = ["#0072B2" if m == "split" else "#999999" for m in modes]
    b1 = ax1.bar(x, cap.loc[modes, "sustained_rps"], color=colors)
    ax1.set_title("Sustained verdicts/s (all flows done ≤10 s)")
    ax2.set_title("Energy per verdict (J), at the sustained rate")
    b2 = ax2.bar(x, en.loc[modes, "j_per_verdict"], color=colors)
    for ax, bars, fmt in ((ax1, b1, "{:.1f}"), (ax2, b2, "{:.2f}")):
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABELS[m] for m in modes], fontsize=8)
        for b in bars:
            ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(D, "edge_capacity.png")
    fig.savefig(out, dpi=200)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
