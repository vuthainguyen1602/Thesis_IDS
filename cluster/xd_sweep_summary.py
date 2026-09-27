#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarise the cross-dataset stability sweep.

``run_xd_seed_sweep.sh`` drops one CSV per repetition into
``results/ml_11_cross_dataset/sweep/``. This script turns that directory into
the numbers the manuscripts quote, so adding repetitions later means re-running
it and copying the rows out. It prints:

  * every run's own values, so the spread is visible
  * mean, sd, coefficient of variation and a Student-t 95% interval per metric
  * the LaTeX rows for the thesis table (comma decimals) and the paper table

Why any of this is needed: the 2017->CSE-CIC-IDS2018 cross-dataset F1 does *not*
reproduce across runs of identical code and configuration, and over 18 runs it
turned out to be bimodal -- 11 runs at 0.0291-0.0843 and 7 at 0.1853-0.3009, with
nothing in between and AUC-PR the same in both groups. In-domain F1 and
cross-dataset AUC-PR stay stable to the third decimal.

So the manuscripts do not quote a mean for that direction: a mean falls in the
empty gap and its interval holds 1 of the 18 runs. They quote the two bands with
their run counts, which is what the LaTeX block below prints for a direction where
this script detects a gap. The other direction is unimodal and keeps mean +/- CI.
The per-seed section exists because the seed does not decide the regime: of five
seeds run twice, three returned to their own regime and two crossed over.

    python3 cluster/xd_sweep_summary.py [sweep_dir]
"""
import csv
import glob
import math
import os
import statistics
import sys

METRICS = ("f1", "precision", "recall", "auc_pr")
# Student-t, two-sided 95%, keyed by SAMPLE SIZE n, so the entry for n already
# accounts for df = n-1. Looking it up at n-1 is an off-by-one that widens every
# interval by one step, which is exactly what happened until n reached 12.
T95_BY_N = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
            8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
            14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
            20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
            26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}


def _t95(n):
    # Two-sided 95% t multiplier for a sample of n values.
    if n <= 1:
        return float("nan")
    if n in T95_BY_N:
        return T95_BY_N[n]
    smaller = [k for k in T95_BY_N if k < n]
    if n > max(T95_BY_N):
        return 1.96
    return T95_BY_N[max(smaller)] if smaller else 12.706


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

    # Where the spread comes from. Two sources are tangled together: the forest
    # seed, and whatever the distributed fit does differently between runs at a
    # fixed seed. Repetitions of one seed measure the second alone, so comparing
    # the two spreads says which one dominates -- the thing a reader will ask.
    by_seed = {}
    for (kind, train, test), runs in groups.items():
        if kind != "cross":
            continue
        for label, vals in runs:
            seed = label.split("seed")[-1] if "seed" in label else label
            by_seed.setdefault(train, {}).setdefault(seed, []).append(vals["f1"])
    for train, seeds in by_seed.items():
        print(f"\n{train} cross-dataset F1 by seed")
        for seed, fs in sorted(seeds.items(), key=lambda kv: -statistics.mean(kv[1])):
            extra = f"   spread {max(fs) - min(fs):.6f}" if len(fs) > 1 else ""
            print(f"   seed {seed:<4} n={len(fs)}  "
                  + " ".join(f"{f:.4f}" for f in fs) + extra)
        within = [max(fs) - min(fs) for fs in seeds.values() if len(fs) > 1]
        means = [statistics.mean(fs) for fs in seeds.values()]
        between = max(means) - min(means) if len(means) > 1 else float("nan")
        if within:
            print(f"   within-seed spread  (same configuration): {max(within):.6f}")
            print(f"   between-seed spread (different seeds)   : {between:.6f}")
            ratio = between / max(within) if max(within) else float("inf")
            print(f"   the seed accounts for {ratio:.0f}x more spread than a repeat does"
                  if ratio > 1 else
                  f"   repeats vary as much as seeds do (ratio {ratio:.2f})")
        else:
            print(f"   between-seed spread: {between:.6f} (no seed was repeated)")

    # A bimodal direction gets bands, not a mean: quoting a mean that lands in the
    # gap misleads, and the outward rounding here is deliberate -- an interval
    # printed with ordinary rounding can exclude the very runs it describes.
    def bands(values):
        v = sorted(values)
        gaps = [(v[i + 1] - v[i], i) for i in range(len(v) - 1)]
        widest, at = max(gaps)
        if widest < 0.25 * (v[-1] - v[0]):
            return None
        lo, hi = v[:at + 1], v[at + 1:]
        return [(len(lo), math.floor(min(lo) * 1e3) / 1e3, math.ceil(max(lo) * 1e3) / 1e3),
                (len(hi), math.floor(min(hi) * 1e3) / 1e3, math.ceil(max(hi) * 1e3) / 1e3)]

    for vn, header in ((True, "thesis (comma decimals)"), (False, "papers (point decimals)")):
        print("\n" + "=" * 72)
        print(f"  LaTeX rows — {header}")
        print("=" * 72)
        for (kind, train, test), st in summary.items():
            f1, pr = st["f1"], st["auc_pr"]
            bd = bands([vals["f1"] for _, vals in groups[(kind, train, test)]]) if kind == "cross" else None
            t = ("cùng bộ" if kind == "in-domain" else "khác bộ") if vn else kind
            if bd:
                cell = " $\\mid$ ".join(
                    (f"{_vn(lo, 3)}--{_vn(hi, 3)} ({n} lượt)" if vn
                     else f"{lo:.3f}--{hi:.3f} ({n})") for n, lo, hi in bd)
            elif kind == "in-domain":
                cell = _vn(f1["mean"]) if vn else f"{f1['mean']:.4f}"
            else:
                cell = (f"{_vn(f1['mean'])} $\\pm$ {_vn(f1['ci'])}" if vn
                        else f"{f1['mean']:.4f} $\\pm$ {f1['ci']:.4f}")
            prc = _vn(pr["mean"]) if vn else f"{pr['mean']:.4f}"
            print(f"{'' if vn else '    '}{train} & {test} & {t} & {cell} & {prc} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
