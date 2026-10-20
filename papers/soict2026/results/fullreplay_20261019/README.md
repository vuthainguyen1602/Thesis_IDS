# Full-replay run for end-to-end detection metrics (2026-10-19)

SOICT reviewer 2 asked for end-to-end precision, recall and F1 after the
anomaly gate. The capacity windows (`../remeasure_20260930/`) each replay the
10,000-flow test sample from row 0, so together they cover only its first
3,601 rows. This run replays the whole sample once through the deployed
pipeline split (gate on Jetson #1, PySpark classifier on Jetson #2).

- `run_fullreplay.sh`: stops k3s on Jetson #1, runs
  `RATE=20 DUR=500 WARM=15 REP=1 NO_ENERGY=1 run_dist_bench.sh split`, starts k3s again.
- `runs/run_split_20261019_083608.json` + clock snapshots (board minus Mac
  +0.050 s before, +0.034 s after; ping 3.6-4.8 ms).
- `fullreplay.csv`: `analyze_runs.py` output (now with tp/fn/fp/tn, precision, F1).

| sent | sustained | drain | e2e p50 / p95 | skip | attack | TP | FN (all gate) | FP | TN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,001 | 19.75/s | 5.9 s | 0.56 / 5.1 s | 89.0% | 1,518 | 1,055 | 463 | 0 | 8,483 | 100% | 69.5% | 0.820 |

10,001 = the sender's loop repeats one row; 10,000 distinct rows. Every missed
attack has route `anomaly_gate_only`; the classifier made no error on the
flows it received.

Pooled over every capacity-window repeat (Table 5 of the manuscript; `pooled_capacity_windows.csv`, written by `../../provenance.py`, also counts the distinct replay rows):
split 19,543 verdicts / 2,990 attack / FN 898 (gate) / FP 0, recall 70.0%;
split_1board (`../journal_20261012/runs`) 5,858 / 882 / 278 / 0, 68.5%;
spark_cluster 2,499 / 381 / 115 / 0, 69.8%; single_gate 1,480 / 200 / 55 / 0, 72.5%;
single + horizontal (gate off) 1,855 / 242 / 0 / 0, 100%.

Reproduce: `python analyze_runs.py runs/run_split_20261019_083608.json --clock-json runs/clock_*`.
