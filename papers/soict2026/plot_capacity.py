#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sustained capacity and energy per verdict per deployment mode (SOICT).

Reads results/remeasure_20260930/capacity_sweep_part{1,2}.csv (analyze_runs.py
rows) and energy_at_sustained_rate.csv. A rate counts as sustained when every
repeat drained within DRAIN_OK seconds; the bar is the mean sustained verdict
rate over the repeats at the highest such rate.

    python papers/soict2026/plot_capacity.py
    python papers/soict2026/plot_capacity.py --vi thesis/img/edge_capacity_vi.png
"""
import argparse
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
LABELS_VI = {"single": "Một nút", "single_gate": "Một nút\n+ cổng",
             "horizontal": "B: Ngang", "spark_cluster": "C: Cụm\nSpark",
             "split": "A: Tách\npipeline"}
TITLES = {"en": ("Sustained verdicts/s (all flows done ≤10 s)",
                 "Energy per verdict (J), at the sustained rate"),
          "vi": ("Phán quyết/giây duy trì được (mọi luồng xong ≤10 s)",
                 "Năng lượng mỗi phán quyết (J) ở tốc độ duy trì")}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--vi", metavar="OUT", help="Vietnamese labels, written to OUT")
    args = ap.parse_args()
    labels, titles = (LABELS_VI, TITLES["vi"]) if args.vi else (LABELS, TITLES["en"])
    cap = capacity_table()
    en = pd.read_csv(os.path.join(D, "energy_at_sustained_rate.csv")).set_index("mode")
    modes = [m for m in ORDER if m in cap.index]
    print(cap.loc[modes].round(2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    x = range(len(modes))
    colors = ["#0072B2" if m == "split" else "#999999" for m in modes]
    b1 = ax1.bar(x, cap.loc[modes, "sustained_rps"], color=colors)
    ax1.set_title(titles[0])
    ax2.set_title(titles[1])
    b2 = ax2.bar(x, en.loc[modes, "j_per_verdict"], color=colors)
    for ax, bars, fmt in ((ax1, b1, "{:.1f}"), (ax2, b2, "{:.2f}")):
        ax.set_xticks(list(x))
        ax.set_xticklabels([labels[m] for m in modes], fontsize=8)
        for b in bars:
            txt = fmt.format(b.get_height())
            ax.annotate(txt.replace(".", ",") if args.vi else txt, (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if args.vi:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    fig.tight_layout()
    out = args.vi or os.path.join(D, "edge_capacity.png")
    fig.savefig(out, dpi=200)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
