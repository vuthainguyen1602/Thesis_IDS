#!/usr/bin/env python
"""Check a demo run's stored verdicts against the CICIDS2017 replay labels.

Every verdict in PostgreSQL carries the replay row of its flow and the send
time (raw_features.row / raw_features.sent_ts); the anomaly gate stores a
benign verdict for each flow it skips (route anomaly_gate_only). Joining the
rows with jetson/sender/replay_cicids2017_labels.csv gives, per run, how many
attack flows were sent, how many were caught and where the rest were lost.

    python demo/verify_run.py --from <send-start epoch> --to <send-end epoch>
"""
import argparse
import csv
import os

import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, "jetson", "sender", "replay_cicids2017_labels.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="t_from", type=float, required=True)
    ap.add_argument("--to", dest="t_to", type=float, required=True)
    args = ap.parse_args()

    # CICIDS2017 ships "Web Attack \ufffd Brute Force" (a broken dash); print it as "-"
    labels = [(r["label"].replace("\ufffd", "-"), int(r["label_binary"]))
              for r in csv.DictReader(open(LABELS, encoding="utf-8", errors="replace"))]
    conn = psycopg2.connect(host="localhost", port=5433, dbname="ids_edge",
                            user="ids", password="ids_password")
    cur = conn.cursor()
    cur.execute("""SELECT prediction, raw_features->>'route', (raw_features->>'row')::int
                   FROM predictions
                   WHERE (raw_features->>'sent_ts')::float BETWEEN %s AND %s""",
                (args.t_from, args.t_to))
    rows = [r for r in cur.fetchall() if r[2] is not None]
    sent = max(r[2] for r in rows) + 1
    skipped = [r for r in rows if r[1] == "anomaly_gate_only"]
    spark = [r for r in rows if r[1] != "anomaly_gate_only"]
    attack = [r for r in rows if labels[r[2]][1] == 1]
    caught = [r for r in attack if r[0] == 1]
    missed_gate = [r for r in attack if r[0] == 0 and r[1] == "anomaly_gate_only"]
    missed_spark = [r for r in attack if r[0] == 0 and r[1] != "anomaly_gate_only"]
    benign = [r for r in rows if labels[r[2]][1] == 0]
    false_alarm = [r for r in benign if r[0] == 1]
    kinds = {}
    for r in attack:
        k = labels[r[2]][0]
        kinds.setdefault(k, [0, 0])
        kinds[k][0] += 1
        kinds[k][1] += r[0] == 1

    w = 46
    print("=" * 60)
    print("  VERIFY RUN AGAINST CICIDS2017 LABELS")
    print("=" * 60)
    print(f"  {'Flows sent (replay rows 0..N-1)':{w}}{sent:>8,}")
    print(f"  {'Verdicts stored in PostgreSQL':{w}}{len(rows):>8,}")
    print(f"  {'  skipped by the gate (benign)':{w}}{len(skipped):>8,}  ({100 * len(skipped) / len(rows):.1f}%)")
    print(f"  {'  classified by PySpark':{w}}{len(spark):>8,}  ({100 * len(spark) / len(rows):.1f}%)")
    print("-" * 60)
    print(f"  {'Attack flows sent (ground truth)':{w}}{len(attack):>8,}")
    print(f"  {'  caught (attack verdict)':{w}}{len(caught):>8,}  -> recall {100 * len(caught) / len(attack):.1f}%")
    print(f"  {'  missed: skipped by the gate':{w}}{len(missed_gate):>8,}")
    print(f"  {'  missed: PySpark said benign':{w}}{len(missed_spark):>8,}")
    print(f"  {'Benign flows sent':{w}}{len(benign):>8,}")
    print(f"  {'  false alarms (attack verdict)':{w}}{len(false_alarm):>8,}  ({100 * len(false_alarm) / max(len(benign), 1):.2f}%)")
    print("-" * 60)
    print("  Attack flows by type (caught / sent)")
    for k, (n, c) in sorted(kinds.items(), key=lambda kv: -kv[1][0]):
        print(f"    {k:{w - 2}}{c:>5} / {n:<5}")
    print("=" * 60)


if __name__ == "__main__":
    main()
