#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Queue-aware analysis of the edge benchmark runs.

benchmark_distributed.py counts the verdicts written *inside* the load window.
When a stage cannot keep up, its verdicts land after the window, so that count
understates the backlog, the gate-skip ratio it implies is inflated, and the
per-node inference p95 hides the time flows spend queued. This script instead
follows every flow *sent* in the load window, using the send timestamp and
replay row that the edge now stores with each verdict:

  send_rps       flows actually sent per second (the sender runs below its target)
  completed      flows sent in the window that received a verdict at all
  e2e p50/p95    verdict time - send time, queueing included (NTP clocks)
  drain_s        last verdict for those flows - end of the load window
  sustained_rps  completed / (last verdict - start of the load window)
  skip_pct       gate-only verdicts / completed (the gate's real skip ratio)
  attack_recall  attack flows given an Attack verdict / attack flows completed

    python papers/soict2026/analyze_runs.py results/benchmarks/run_split_rep*_20260930_*.json
"""

import argparse
import csv
import json
import os
import statistics

import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "..", "..", "jetson", "sender", "replay_cicids2017_labels.csv")


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def analyze(run_path, cur, labels):
    run = json.load(open(run_path))
    start, end = run["load_start_epoch"], run["load_end_epoch"]
    cur.execute(
        """SELECT timestamp, prediction, raw_features->>'route',
                  (raw_features->>'sent_ts')::float, (raw_features->>'row')::int
           FROM predictions
           WHERE (raw_features->>'sent_ts')::float BETWEEN %s AND %s""",
        (start, end),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    e2e = [(ts - sent) * 1000.0 for ts, _, _, sent, _ in rows]
    e2e_clf = [(ts - sent) * 1000.0 for ts, _, route, sent, _ in rows if route != "anomaly_gate_only"]
    last = max(r[0] for r in rows)
    gate_only = sum(1 for r in rows if r[2] == "anomaly_gate_only")
    attacks = [r for r in rows if r[4] is not None and labels[r[4]] == 1]
    caught = sum(1 for r in attacks if r[1] == 1)
    # The sender restarts its row counter at 0 for every send, so the rows sent in
    # this window are 0..max; its sleep-paced loop runs below the requested rate.
    sent = max(r[4] for r in rows if r[4] is not None) + 1
    return {
        "mode": run.get("deploy_mode"),
        "load_start": round(start),
        "repeat": run.get("repeat_index"),
        "sent": sent,
        "send_rps": round(sent / (end - start), 1),
        "completed": len(rows),
        "completed_pct": round(100.0 * len(rows) / sent, 1),
        "in_window_rps": round(sum(1 for r in rows if r[0] <= end) / (end - start), 2),
        "sustained_rps": round(len(rows) / (last - start), 2),
        "drain_s": round(max(0.0, last - end), 1),
        "e2e_p50_ms": round(pct(e2e, 0.50), 1),
        "e2e_p95_ms": round(pct(e2e, 0.95), 1),
        "e2e_p95_classified_ms": round(pct(e2e_clf, 0.95), 1) if e2e_clf else None,
        "skip_pct": round(100.0 * gate_only / len(rows), 1),
        "attack_flows": len(attacks),
        "attack_recall_pct": round(100.0 * caught / len(attacks), 1) if attacks else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--rate", type=float, default=100.0, help="requested sender rate, recorded in the CSV")
    ap.add_argument("--csv", default=None, help="append per-repeat rows to this CSV")
    args = ap.parse_args()

    labels = [int(r["label_binary"]) for r in csv.DictReader(open(LABELS))]
    conn = psycopg2.connect(host="localhost", port=5433, dbname="ids_edge", user="ids", password="ids_password")
    cur = conn.cursor()
    results = [r for r in (analyze(p, cur, labels) for p in sorted(args.runs)) if r]
    for r in results:
        print(r)
    if len(results) > 1:
        keys = ["sustained_rps", "e2e_p95_ms", "drain_s", "skip_pct", "attack_recall_pct"]
        print("mean:", {k: round(statistics.mean(r[k] for r in results if r[k] is not None), 2) for k in keys})
    if args.csv and results:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()) + ["rate"])
            if new:
                w.writeheader()
            for r in results:
                w.writerow({**r, "rate": args.rate})


if __name__ == "__main__":
    main()
