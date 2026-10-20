#!/usr/bin/env python
"""Write the source tables for SOICT numbers that were only stated in READMEs.

    results/cicids2018_replay_20261019/classifier_alone.csv
        the deployed forest (ONNX export, identical predictions to PySpark) on each
        10,000-flow replay sample without the gate: Table 6 "classifier alone" row
    results/fullreplay_20261019/pooled_capacity_windows.csv
        Table 5 capacity-window rows: every repeat of a mode pooled, with the number
        of distinct replay rows they cover (needs the edge Postgres on :5433)
    results/journal_20261012/ping_rtt.csv
        Mac -> board ping RTT from every clock snapshot (Discussion: 4-9 ms)

    python papers/soict2026/provenance.py
"""
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(HERE, "results")
MODEL = os.path.join(ROOT, "jetson", "model")
SENDER = os.path.join(ROOT, "jetson", "sender")


def classifier_alone():
    import onnxruntime as ort
    cols = json.load(open(os.path.join(MODEL, "feature_columns.json")))
    sess = ort.InferenceSession(os.path.join(MODEL, "ids_rf.onnx"))
    name = sess.get_inputs()[0].name
    rows = []
    for ds in ("cicids2017", "cicids2018"):
        x = pd.read_csv(os.path.join(SENDER, f"replay_{ds}.csv"))[cols].values.astype(np.float32)
        y = pd.read_csv(os.path.join(SENDER, f"replay_{ds}_labels.csv"))["label_binary"].values.astype(int)
        p = sess.run(None, {name: x})[0].astype(int).ravel()
        tp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
        fp, tn = int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
        rows.append({"replay": ds, "flows": len(y), "attack": tp + fn, "tp": tp, "fn": fn,
                     "fp": fp, "tn": tn, "recall_pct": round(100 * tp / (tp + fn), 2)})
    out = os.path.join(R, "cicids2018_replay_20261019", "classifier_alone.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def pooled_windows():
    import psycopg2
    groups = {
        "A: capacity windows": [f"{R}/remeasure_20260930/runs/run_split_rep*_edge-node-1_*.json"],
        "A1: split, one board": [f"{R}/journal_20261012/runs/run_split_1board_rep*_*.json"],
        "C: Spark cluster": [f"{R}/remeasure_20260930/runs/run_spark_cluster_rep*_edge-node-1_*.json"],
        "Single node + gate": [f"{R}/remeasure_20260930/runs/run_single_gate_rep*_edge-node-1_*.json"],
        "Gate off (single, B)": [f"{R}/remeasure_20260930/runs/run_single_rep*_edge-node-1_*.json",
                                 f"{R}/remeasure_20260930/runs/run_horizontal_rep*_edge-node-1_*.json"],
    }
    conn = psycopg2.connect(host="localhost", port=5433, dbname="ids_edge", user="ids", password="ids_password")
    cur = conn.cursor()
    rows = []
    for mode, pats in groups.items():
        runs = sorted(p for pat in pats for p in glob.glob(pat))
        with tempfile.TemporaryDirectory() as t:
            per = os.path.join(t, "runs.csv")
            subprocess.run([sys.executable, os.path.join(HERE, "analyze_runs.py"), *runs, "--csv", per],
                           check=True, capture_output=True)
            a = list(csv.DictReader(open(per)))
        distinct = set()
        for p in runs:
            run = json.load(open(p))
            cur.execute("SELECT DISTINCT (raw_features->>'row')::int FROM predictions "
                        "WHERE (raw_features->>'sent_ts')::float BETWEEN %s AND %s",
                        (run["load_start_epoch"], run["load_end_epoch"]))
            distinct |= {r[0] for r in cur.fetchall() if r[0] is not None}
        s = lambda k: sum(int(x[k]) for x in a)
        tp, fn, fp, att = s("tp"), s("fn"), s("fp"), s("attack_flows")
        rows.append({"mode": mode, "runs": len(a), "verdicts": s("completed"), "attack": att,
                     "tp": tp, "fn": fn, "fp": fp,
                     "recall_pct": round(100 * tp / att, 1),
                     "f1": round(2 * tp / (2 * tp + fn + fp), 3),
                     "distinct_rows": len(distinct)})
    out = os.path.join(R, "fullreplay_20261019", "pooled_capacity_windows.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def ping_rtt():
    files = sorted(glob.glob(f"{R}/journal_20261012/clock_*.json")
                   + glob.glob(f"{R}/journal_20261012/runs/clock_*.json")
                   + glob.glob(f"{R}/fullreplay_20261019/runs/clock_*.json")
                   + glob.glob(f"{R}/cicids2018_replay_20261019/runs/clock_*.json"))
    rows = []
    for f in files:
        d = json.load(open(f))
        for h in d.get("hosts", [d]):
            if h.get("ping_rtt_avg_ms") is None:
                continue
            rows.append({"file": os.path.relpath(f, R), "host": h["host"],
                         "ping_count": h.get("ping_count"), "rtt_min_ms": h["ping_rtt_min_ms"],
                         "rtt_avg_ms": h["ping_rtt_avg_ms"], "rtt_max_ms": h["ping_rtt_max_ms"],
                         "loss_pct": h.get("ping_loss_pct")})
    out = os.path.join(R, "journal_20261012", "ping_rtt.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    for fn in (classifier_alone, pooled_windows, ping_rtt):
        print(f"[OK] {fn()}")
