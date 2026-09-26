#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarise the cross-dataset stability sweep.

``run_xd_seed_sweep.sh`` drops one CSV per repetition into
``results/ml_11_cross_dataset/sweep/``. This script turns that directory into
the numbers the manuscripts quote, so adding repetitions later is a re-run of
this script and a copy-paste, not a hand computation:

  * per-run values, to see the spread rather than a summary of it
  * mean, sd, coefficient of variation and a Student-t 95% interval per metric
  * the LaTeX rows for the thesis table (comma decimals) and the paper table

Why an interval at all: the 2017->CSE-CIC-IDS2018 cross-dataset F1 is *not*
reproducible across runs of identical code and configuration (two runs with the
same seed disagreed), while in-domain F1 and cross-dataset AUC-PR are stable to
the third decimal. A single draw from that distribution is not a reportable
number; the mean over repetitions with its interval is.

    python3 cluster/xd_sweep_summary.py [sweep_dir]
"""
import csv
import glob
import os
import statistics
import sys

METRICS = ("f1", "precision", "recall", "auc_pr")
# Student-t, two-sided 95%, df = n-1. Normal-approximation z for large n.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
       9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 15: 2.145, 20: 2.093}


def _t95(n):
    return T95.get(n - 1, 1.96) if n > 1 else float("nan")


def _vn(x, digits=4):
    return f"{x:.{digits}f}".replace(".", "{,}")


def load(sweep_dir):
    """{(kind, train, test): [(run_label, {metric: value}), ...]}"""
    groups = {}
    for path in sorted(glob.glob(os.path.join(sweep_dir, "*.csv"))):
        label = os.path.basename(path)[:-4]
        with open(path) as fh:
            for row in csv.DictReader(fh):
                key = (row["kind"], row["train"], row["test"])
                vals = {m: float(row[m]) for m in METRICS if row.get(m) not in (None, "")}
                groups.setdefault(key, []).append((label, vals))
    return groups


def stats(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    ci = _t95(n) * sd / n ** 0.5 if n > 1 else float("nan")
    cv = 100.0 * sd / mean if mean else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci": ci, "cv": cv,
            "min": min(values), "max": max(values)}


def main():
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        "results", "ml_11_cross_dataset", "sweep")
    groups = load(sweep_dir)
    if not groups:
        print(f"[ERR] no sweep CSVs in {sweep_dir}")
        return 1

    summary = {}
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        kind, train, test = key
        runs = groups[key]
        print(f"\n{train} -> {test}   [{kind}]   n={len(runs)}")
        header = "   " + f"{'run':16s}" + "".join(f"{m:>11s}" for m in METRICS)
        print(header)
        for label, vals in runs:
            print("   " + f"{label:16s}"
                  + "".join(f"{vals.get(m, float('nan')):11.4f}" for m in METRICS))
        summary[key] = {m: stats([v[m] for _, v in runs if m in v]) for m in METRICS}
        for line, fmt in (("mean", "{0[mean]:11.4f}"), ("sd", "{0[sd]:11.4f}"),
                          ("95% CI +/-", "{0[ci]:11.4f}"), ("CV %", "{0[cv]:11.1f}")):
            print("   " + f"{line:16s}"
                  + "".join(fmt.format(summary[key][m]) for m in METRICS))

    # Determinism check: repetitions that share a seed must agree bit-for-bit.
    same_seed = {}
    for (kind, train, test), runs in groups.items():
        if kind != "cross":
            continue
        for label, vals in runs:
            seed = label.split("seed")[-1] if "seed" in label else label
            same_seed.setdefault((train, seed), []).append(vals["f1"])
    repeated = {k: v for k, v in same_seed.items() if len(v) > 1}
    if repeated:
        spread = max(max(v) - min(v) for v in repeated.values())
        print(f"\nRepeated-seed spread: {spread:.6f} -> "
              + ("NOT deterministic (identical configuration, different results)"
                 if spread > 1e-9 else "deterministic under a fixed seed"))

    print("\n" + "=" * 72)
    print("  LaTeX rows — thesis (comma decimals), F1 as mean +/- CI over n runs")
    print("=" * 72)
    for (kind, train, test), st in summary.items():
        t = "cùng bộ" if kind == "in-domain" else "khác bộ"
        f1, pr = st["f1"], st["auc_pr"]
        cell = (f"{_vn(f1['mean'])}" if kind == "in-domain"
                else f"{_vn(f1['mean'])} $\\pm$ {_vn(f1['ci'])}")
        print(f"{train} & {test} & {t} & {cell} & {_vn(pr['mean'])} \\\\")
    print("\n" + "=" * 72)
    print("  LaTeX rows — papers (point decimals)")
    print("=" * 72)
    for (kind, train, test), st in summary.items():
        f1, pr = st["f1"], st["auc_pr"]
        cell = (f"{f1['mean']:.4f}" if kind == "in-domain"
                else f"{f1['mean']:.4f} $\\pm$ {f1['ci']:.4f}")
        print(f"    {train} & {test} & {kind} & {cell} & {pr['mean']:.4f} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
