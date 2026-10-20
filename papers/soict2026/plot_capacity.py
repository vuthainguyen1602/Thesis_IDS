#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sustained capacity and energy per verdict per deployment mode (SOICT).

Reads results/remeasure_20260930/capacity_sweep_part{1,2}.csv (analyze_runs.py
rows) and energy_at_sustained_rate.csv, plus the one-board split (A1) from
results/journal_20261012/ (same rule, clock-corrected tables). A rate counts as sustained when every
repeat drained within DRAIN_OK seconds; the bar is the mean sustained verdict
rate over the repeats at the highest such rate.

    python papers/soict2026/plot_capacity.py
    python papers/soict2026/plot_capacity.py --vi thesis/img/edge_capacity_vi.png
"""
import argparse
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3, in the PDF copy
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "remeasure_20260930")
J = os.path.join(HERE, "results", "journal_20261012")
DRAIN_OK = 10.0
ORDER = ["single", "single_gate", "horizontal", "spark_cluster", "split_1board", "split"]
LABELS = {"single": "Single node", "single_gate": "Single node\n+ gate",
          "horizontal": "B:\nHorizontal", "spark_cluster": "C: Spark\ncluster",
          "split_1board": "A1: Split,\none board", "split": "A: Pipeline\nsplit"}
# Shorter labels for the print-size (camera-ready) PDF.
PRINT_LABELS = {"single": "Single\nnode", "single_gate": "Single\n+ gate", "horizontal": "B:\nhoriz.",
                "spark_cluster": "C:\ncluster", "split_1board": "A1:\n1-board", "split": "A:\n2-board"}
LABELS_VI = {"single": "Một nút", "single_gate": "Một nút\n+ cổng",
             "horizontal": "B: Ngang", "spark_cluster": "C: Cụm\nSpark",
             "split_1board": "A1: Tách,\ncùng một bo", "split": "A: Tách\npipeline"}
TITLES = {"en": ("Sustained verdicts/s (all flows done ≤10 s)",
                 "Energy per verdict (J), at the sustained rate"),
          "vi": ("Dự đoán/giây duy trì được (mọi luồng xong ≤10 s)",
                 "Năng lượng mỗi dự đoán (J) ở tốc độ duy trì")}


def half_up(v: float, nd: int) -> float:
    """Round half up on the decimal value: format() rounds the binary float, so
    the single+gate mean (1.92 + 1.98) / 2 = 1.95 printed as 1.9."""
    return float(Decimal(repr(round(v, 6))).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP))


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


def one_board_split():
    """A1 capacity and energy rows from the journal sweep tables."""
    cap = pd.read_csv(os.path.join(J, "capacity_summary.csv"))
    cap = cap[(cap["mode"] == "split_1board") & (cap["sustained"] == 1)]
    top = cap[cap["rate"] == cap["rate"].max()].iloc[0]
    en = pd.read_csv(os.path.join(J, "energy_at_sustained_rate.csv")).set_index("mode")
    return top["sustained_rps"], en.loc["split_1board", "j_per_verdict"]


def draw(cap, en, modes, labels, titles, vi, figsize, fs, title_fs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = range(len(modes))
    colors = ["#0072B2" if m == "split" else "#56B4E9" if m == "split_1board" else "#999999" for m in modes]
    b1 = ax1.bar(x, cap.loc[modes, "sustained_rps"], color=colors)
    title_kw = {"fontsize": title_fs} if title_fs else {}
    ax1.set_title(titles[0], **title_kw)
    ax2.set_title(titles[1], **title_kw)
    b2 = ax2.bar(x, en.loc[modes, "j_per_verdict"], color=colors)
    lower_bound = {"spark_cluster"}  # the next rate step had one of two repeats keep up
    for ax, bars, fmt in ((ax1, b1, "{:.1f}"), (ax2, b2, "{:.2f}")):
        ax.set_xticks(list(x))
        ax.set_xticklabels([labels[m] for m in modes], fontsize=fs)
        if title_fs:
            ax.tick_params(axis="y", labelsize=fs)
        for m, b in zip(modes, bars):
            txt = fmt.format(half_up(b.get_height(), 1 if ax is ax1 else 2))
            if ax is ax1 and m in lower_bound:
                txt = "≥" + txt
            ax.annotate(txt.replace(".", ",") if vi else txt, (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=fs)
        ax.spines[["top", "right"]].set_visible(False)
        if vi:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    fig.tight_layout(pad=0.3 if title_fs else 1.08)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vi", metavar="OUT", help="Vietnamese labels, written to OUT")
    args = ap.parse_args()
    labels, titles = (LABELS_VI, TITLES["vi"]) if args.vi else (LABELS, TITLES["en"])
    cap = capacity_table()
    en = pd.read_csv(os.path.join(D, "energy_at_sustained_rate.csv")).set_index("mode")
    cap.loc["split_1board", "sustained_rps"], en.loc["split_1board", "j_per_verdict"] = one_board_split()
    modes = [m for m in ORDER if m in cap.index]
    print(cap.loc[modes].round(2))

    out = args.vi or os.path.join(D, "edge_capacity.png")
    draw(cap, en, modes, labels, titles, args.vi, (10, 3.4), 8, None).savefig(out, dpi=200)
    if not args.vi:
        # Vector copy for the Springer camera-ready, drawn at the LNCS text width
        # (4.8 in) so it is included 1:1 and no lettering falls below 6 pt.
        draw(cap, en, modes, PRINT_LABELS, titles, False, (4.8, 1.75), 6, 7).savefig(out[:-4] + ".pdf")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
