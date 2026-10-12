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


def clock_offset_from_json(paths):
    """Mean board-minus-sender clock offset (s) over the given clock_offset.py
    snapshots; each file holds one host or a {"hosts": [...]} list."""
    offsets = []
    for p in paths:
        data = json.load(open(p))
        for h in data.get("hosts", [data]):
            if h.get("offset_s") is not None:
                offsets.append(float(h["offset_s"]))
    if offsets and max(offsets) - min(offsets) > 0.05:
        print(f"[WARN] clock offsets spread {max(offsets) - min(offsets):.3f} s across "
              f"{len(offsets)} snapshots; a clock was adjusted during the run?")
    return statistics.mean(offsets) if offsets else 0.0


def run_clock_offset(run):
    """Offset from the snapshots the run driver took right before and after
    this load window (benchmark_distributed.py >= 2026-10-12), else None."""
    offsets = []
    for key in ("clock_before", "clock_after"):
        snap = run.get(key) or {}
        offsets += [float(h["offset_s"]) for h in snap.get("hosts", []) if h.get("offset_s") is not None]
    if not offsets:
        return None, None
    return statistics.mean(offsets), max(offsets) - min(offsets)


def analyze(run_path, cur, labels, clock_offset_s=0.0):
    run = json.load(open(run_path))
    start, end = run["load_start_epoch"], run["load_end_epoch"]
    own_offset, own_spread = run_clock_offset(run)
    if own_offset is not None:
        clock_offset_s = own_offset
        if own_spread > 0.05:
            print(f"[WARN] {os.path.basename(run_path)}: clock stepped {own_spread:.3f} s "
                  "between the snapshots around this window; re-run it")
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
    # Verdict timestamps come from the board's clock, send timestamps from the
    # sender's; clock_offset_s (board minus sender, from clock_offset.py) is
    # taken off the verdict side before any latency is formed.
    rows = [(ts - clock_offset_s, pred, route, sent, row) for ts, pred, route, sent, row in rows]
    e2e = [(ts - sent) * 1000.0 for ts, _, _, sent, _ in rows]
    e2e_clf = [(ts - sent) * 1000.0 for ts, _, route, sent, _ in rows if route != "anomaly_gate_only"]
    last = max(r[0] for r in rows)
    gate_only = sum(1 for r in rows if r[2] == "anomaly_gate_only")
    attacks = [r for r in rows if r[4] is not None and labels[r[4]] == 1]
    caught = sum(1 for r in attacks if r[1] == 1)
    # Rows sent: the sender reports its count since 2026-10-12 (it may loop the
    # CSV, so row indexes repeat). Older runs: the sender restarted its row
    # counter at 0 for every send, so the rows sent in the window are 0..max.
    sent = (run.get("send") or {}).get("rows_sent")
    if not sent:
        sent = max(r[4] for r in rows if r[4] is not None) + 1
    return {
        "mode": run.get("deploy_mode"),
        "load_start": round(start),
        "repeat": run.get("repeat_index"),
        "clock_offset_s": round(clock_offset_s, 4),
        "clock_spread_s": round(own_spread, 4) if own_spread is not None else None,
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
    ap.add_argument("--clock-offset-s", type=float, default=None,
                    help="board clock minus sender clock, subtracted from verdict timestamps")
    ap.add_argument("--clock-json", nargs="*", default=[],
                    help="jetson/scripts/clock_offset.py snapshots; their mean offset is used "
                         "(give the run files first, this option last)")
    args = ap.parse_args()

    offset = args.clock_offset_s
    if offset is None:
        offset = clock_offset_from_json(args.clock_json) if args.clock_json else 0.0
    if offset:
        print(f"clock offset applied: {offset:+.4f} s (board minus sender)")

    labels = [int(r["label_binary"]) for r in csv.DictReader(open(LABELS))]
    conn = psycopg2.connect(host="localhost", port=5433, dbname="ids_edge", user="ids", password="ids_password")
    cur = conn.cursor()
    results = [r for r in (analyze(p, cur, labels, offset) for p in sorted(args.runs)) if r]
    for r in results:
        print(r)
    if len(results) > 1:
        keys = ["sustained_rps", "e2e_p95_ms", "drain_s", "skip_pct", "attack_recall_pct"]
        print("mean:", {k: round(statistics.mean(r[k] for r in results if r[k] is not None), 2) for k in keys})
    if args.csv and results:
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        fieldnames = list(results[0].keys()) + ["rate"]
        if not new:
            # Keep the existing header; columns added later stay in the run JSONs.
            fieldnames = next(csv.reader(open(args.csv)))
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in results:
                w.writerow({**r, "rate": args.rate})


if __name__ == "__main__":
    main()
