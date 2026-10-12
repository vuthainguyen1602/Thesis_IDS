#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Energy per verdict at the sustained rate, one row per mode.

Same procedure as results/remeasure_20260930/energy_at_sustained_rate.csv, now
automated: for every energy run ``run_<mode>_<T>.json`` in the runs directory
(the file without ``_rep`` that run_dist_bench.sh writes when the energy window
is on), sum the participating boards' ``power_<node>_<T>.json`` and divide by
the verdict rate of that run's load window, taken from the analyze_runs.py row
in energy_runs.csv (clock-corrected end-to-end latency and drain come along).

    python energy_summary.py results/journal_20261012/runs \
        --out results/journal_20261012/energy_at_sustained_rate.csv

  j_per_verdict         = sum(avg_power_w) / send_rps          (whole board(s))
  active_mj_per_verdict = sum(active_power_w) / send_rps * 1e3 (idle subtracted)
"""

import argparse
import csv
import glob
import json
import os
import re

BOARDS = {
    "single": 1, "single_gate": 1, "single_onnx": 1, "single_gate_onnx": 1,
    "split_1board": 1, "split_1board_onnx": 1,
    "split": 2, "split_onnx": 2, "spark_cluster": 2,
    "horizontal": 2, "horizontal_gate": 2, "horizontal_onnx": 2, "horizontal_gate_onnx": 2,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_dir")
    ap.add_argument("--analysis", default=None, help="energy_runs.csv (default: <runs_dir>/energy_runs.csv)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    analysis = {}
    for r in csv.DictReader(open(args.analysis or os.path.join(args.runs_dir, "energy_runs.csv"))):
        analysis[(r["mode"], int(float(r["load_start"])))] = r

    rows = []
    for run_path in sorted(glob.glob(os.path.join(args.runs_dir, "run_*.json"))):
        name = os.path.basename(run_path)
        if "_rep" in name:
            continue
        m = re.match(r"run_(.+)_(\d{8}_\d{6})\.json$", name)
        if not m:
            continue
        mode, tag = m.groups()
        powers = sorted(glob.glob(os.path.join(args.runs_dir, f"power_*_{tag}.json")))
        if not powers:
            print(f"[skip] {name}: no power_*_{tag}.json")
            continue
        run = json.load(open(run_path))
        key = (mode, round(run["load_start_epoch"]))
        a = analysis.get(key)
        if a is None:
            print(f"[skip] {name}: no analyze_runs row for load_start {key[1]}")
            continue
        p = [json.load(open(x)) for x in powers]
        avg_w = sum(x["avg_power_w"] for x in p)
        act_w = sum(x["active_power_w"] for x in p)
        rps = float(a["send_rps"])
        rows.append({
            "mode": mode, "boards": BOARDS.get(mode, len(p)), "power_files": len(p),
            "send_rps": rps, "sustained_rps": a["sustained_rps"], "drain_s": a["drain_s"],
            "avg_power_w": round(avg_w, 2), "active_power_w": round(act_w, 2),
            "j_per_verdict": round(avg_w / rps, 3) if rps else None,
            "active_mj_per_verdict": round(act_w / rps * 1000, 1) if rps else None,
            "e2e_p50_ms": a["e2e_p50_ms"], "e2e_p95_ms": a["e2e_p95_ms"],
            "skip_pct": a["skip_pct"], "attack_recall_pct": a["attack_recall_pct"],
            "clock_offset_s": a.get("clock_offset_s"), "run": name,
        })
        print(rows[-1])

    if args.out and rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
