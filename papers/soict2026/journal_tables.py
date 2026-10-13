#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final tables for the journal edge runs (results/journal_<stamp>/).

Re-analyses every run JSON in the runs directory against Postgres with the
clock offset that belongs to that run (the snapshots taken around its own load
window when the driver stored them, else the mode-level before/after pair of
the orchestrator invocation that produced it), flags runs whose window
straddled a step of the sender's clock (offset spread > 50 ms), and writes:

  capacity_all.csv      one row per repeat (+ clock_offset_s, clock_spread_s, tainted)
  capacity_summary.csv  one row per (mode, rate): means over the clean repeats
  energy_at_sustained_rate.csv  one row per mode from the latest clean energy run

    python journal_tables.py results/journal_20261012/runs
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import statistics
import subprocess
import sys

import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_runs import LABELS, analyze  # noqa: E402
from energy_summary import BOARDS  # noqa: E402

SPREAD_OK = 0.05


def offsets(snap):
    return [float(h["offset_s"]) for h in (snap or {}).get("hosts", []) if h.get("offset_s") is not None]


def mode_level_clocks(runs_dir):
    """{(mode, T_epoch): [offsets before+after]} from clock_<mode>_<T>_{before,after}.json."""
    out = {}
    for f in glob.glob(os.path.join(runs_dir, "clock_*_before.json")):
        m = re.match(r"clock_(.+)_(\d{8}_\d{6})_before\.json$", os.path.basename(f))
        if not m:
            continue
        mode, T = m.groups()
        t_epoch = datetime.datetime.strptime(T, "%Y%m%d_%H%M%S").timestamp()
        offs = offsets(json.load(open(f)))
        after = f.replace("_before.json", "_after.json")
        if os.path.exists(after):
            offs += offsets(json.load(open(after)))
        out[(mode, t_epoch)] = offs
    return out


def run_offsets(run, clocks):
    offs = offsets(run.get("clock_before")) + offsets(run.get("clock_after"))
    if offs:
        return offs, "run"
    mode, ls = run["deploy_mode"], run["load_start_epoch"]
    cands = [(t, o) for (m, t), o in clocks.items() if m == mode and t <= ls]
    if not cands:
        return [], "none"
    return max(cands)[1], "mode"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_dir")
    ap.add_argument("--out-dir", default=None, help="default: parent of runs_dir")
    args = ap.parse_args()
    runs_dir = args.runs_dir
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(runs_dir))
    clocks = mode_level_clocks(runs_dir)
    labels = [int(r["label_binary"]) for r in csv.DictReader(open(LABELS))]
    conn = psycopg2.connect(host="localhost", port=5433, dbname="ids_edge", user="ids", password="ids_password")
    cur = conn.cursor()

    reps, energy = [], []
    for path in sorted(glob.glob(os.path.join(runs_dir, "run_*.json"))):
        run = json.load(open(path))
        offs, src = run_offsets(run, clocks)
        if not offs:
            print(f"[skip] {os.path.basename(path)}: no clock data")
            continue
        offset = statistics.mean(offs)
        spread = max(offs) - min(offs)
        row = analyze(path, cur, labels, offset)
        if row is None:
            print(f"[skip] {os.path.basename(path)}: no verdicts")
            continue
        row.update({
            "clock_offset_s": round(offset, 4), "clock_spread_s": round(spread, 4),
            "clock_source": src, "tainted": int(spread > SPREAD_OK),
            "rate": run.get("send", {}).get("requested_rate_rps"),
            "run": os.path.basename(path),
        })
        (reps if "_rep" in os.path.basename(path) else energy).append(row)

    # ---- capacity: every repeat, then clean means per (mode, rate) ----
    fields = list(reps[0].keys())
    with open(os.path.join(out_dir, "capacity_all.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(reps)
    groups = {}
    for r in reps:
        groups.setdefault((r["mode"], float(r["rate"])), []).append(r)
    summary = []
    keys = ["send_rps", "sustained_rps", "e2e_p50_ms", "e2e_p95_ms", "skip_pct", "attack_recall_pct"]
    for (mode, rate), rs in sorted(groups.items()):
        clean = [r for r in rs if not r["tainted"]]
        use = clean or rs
        s = {"mode": mode, "boards": BOARDS.get(mode), "rate": rate,
             "n_repeats": len(rs), "n_clean": len(clean), "latency_from": "clean" if clean else "TAINTED"}
        for k in keys:
            vals = [float(r[k]) for r in use if r[k] is not None]
            s[k] = round(statistics.mean(vals), 1) if vals else None
        # Throughput, drain, skip and recall do not depend on the clock beyond
        # +/-0.5 s on drain, so the saturation rule uses every repeat.
        s["drain_s_max"] = max(float(r["drain_s"]) for r in rs)
        s["sustained"] = int(s["drain_s_max"] <= 10.0)
        summary.append(s)
    with open(os.path.join(out_dir, "capacity_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print(f"[OK] {len(reps)} repeats ({sum(r['tainted'] for r in reps)} tainted) -> capacity_all.csv, "
          f"{len(summary)} (mode, rate) points -> capacity_summary.csv")
    for s in summary:
        print(f"  {s['mode']:22s} @{s['rate']:5.0f}: sustained={s['sustained']} drain_max={s['drain_s_max']:5.1f}s "
              f"clean={s['n_clean']}/{s['n_repeats']} p50={s['e2e_p50_ms']} p95={s['e2e_p95_ms']}")

    # ---- energy: latest clean energy run per mode, with its power files ----
    energy_rows = []
    for row in sorted(energy, key=lambda r: r["load_start"]):
        tag = re.search(r"_(\d{8}_\d{6})\.json$", row["run"]).group(1)
        powers = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(runs_dir, f"power_*_{tag}.json")))]
        if not powers:
            print(f"[skip] {row['run']}: no power files")
            continue
        avg_w = sum(p["avg_power_w"] for p in powers)
        act_w = sum(p["active_power_w"] for p in powers)
        rps = float(row["send_rps"])
        energy_rows.append({
            "mode": row["mode"], "boards": BOARDS.get(row["mode"]), "rate": row["rate"],
            "send_rps": rps, "sustained_rps": row["sustained_rps"], "drain_s": row["drain_s"],
            "avg_power_w": round(avg_w, 2), "active_power_w": round(act_w, 2),
            "j_per_verdict": round(avg_w / rps, 3), "active_mj_per_verdict": round(act_w / rps * 1000, 1),
            "e2e_p50_ms": row["e2e_p50_ms"], "e2e_p95_ms": row["e2e_p95_ms"],
            "skip_pct": row["skip_pct"], "attack_recall_pct": row["attack_recall_pct"],
            "clock_offset_s": row["clock_offset_s"], "clock_spread_s": row["clock_spread_s"],
            "tainted": row["tainted"], "power_files": len(powers), "run": row["run"],
        })
    with open(os.path.join(out_dir, "energy_all.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(energy_rows[0].keys())); w.writeheader(); w.writerows(energy_rows)
    chosen = {}
    for r in energy_rows:          # later runs override earlier ones; clean beats tainted
        prev = chosen.get(r["mode"])
        if prev is None or prev["tainted"] >= r["tainted"]:
            chosen[r["mode"]] = r
    with open(os.path.join(out_dir, "energy_at_sustained_rate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(energy_rows[0].keys())); w.writeheader()
        w.writerows(chosen[m] for m in sorted(chosen))
    print(f"[OK] {len(energy_rows)} energy runs -> energy_all.csv; {len(chosen)} modes -> energy_at_sustained_rate.csv")
    for m in sorted(chosen):
        r = chosen[m]
        print(f"  {m:22s} @{r['rate']:5.0f}: {r['avg_power_w']:5.2f} W, {r['j_per_verdict']:.3f} J/verdict, "
              f"p50={r['e2e_p50_ms']} p95={r['e2e_p95_ms']} drain={r['drain_s']} tainted={r['tainted']}")


if __name__ == "__main__":
    main()
