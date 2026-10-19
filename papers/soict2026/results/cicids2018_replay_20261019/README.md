# CSE-CIC-IDS2018 replay through the deployed pipeline (2026-10-19)

SOICT reviewer 1 asked for a second dataset. This folder replays CSE-CIC-IDS2018
flows through the deployed pipeline split (gate on Jetson #1, PySpark
classifier on Jetson #2), with the models trained on CICIDS2017 left as they are.

**Replay sample.** `jetson/scripts/make_replay_csv.py --test data_2018/test_data.parquet`
draws 10,000 flows (seed 42) from the deduplicated, leak-free 2018 test split
(the 20% stratified sample used by ml_11, columns harmonised to the 2017
names): 12.8% attacks. CICIDS2017 stores "Fwd Header Length" twice
(fwd_header_length34/55, equal on every row); both are filled from the single
2018 column. Output: `jetson/sender/replay_cicids2018{,_labels}.csv`.

**Runs** (`run_2018.sh`; k3s on Jetson #1 stopped for the measurements):
capacity sweep at 5, 8, 12 flows/s (2 repeats, 45 s windows,
`runs/capacity_sweep.csv`), then one run replaying all 10,000 flows at the highest
sustained rate (8 flows/s, 1,250 s; `fullreplay.csv`).

| | CICIDS2017 (`../fullreplay_20261019`) | CSE-CIC-IDS2018 |
|---|---|---|
| gate forwards (skip) | 11.0% (89.0%) | 30.4% (69.6%) |
| benign flows the gate forwards | 0.6% | 28.4% |
| sustained capacity | 22.4 flows/s | ~8 (12: drain 18.5-28.9 s) |
| full replay: rate / drain / e2e p50, p95 | 20/s, 5.9 s, 0.56 / 5.1 s | 8/s, 1.9 s, 0.86 / 7.8 s |
| attack flows | 1,518 | 1,276 |
| TP / FN gate / FN classifier / FP | 1,055 / 463 / 0 / 0 | 10 / 712 / 554 / 0 |
| precision / recall / F1 | 100% / 69.5% / 0.820 | 100% / 0.8% / 0.016 |

Offline check on the same 10,000 flows (gate scorer + `ids_rf.onnx`, which agrees
with the PySpark forest on every prediction) gives the same gate+classifier
counts on both datasets; the classifier alone, without the gate, catches 23 of
1,276 2018 attacks (recall 1.8%, 17 FP).

Per 2018 attack class, the gate forwards: Bot 95%, LOIC-HTTP 53%, HOIC 57%,
Infilteration 33%, Hulk 1%, SSH-Bruteforce 0%, GoldenEye 5%.

Reproduce: `REPLAY_LABELS=jetson/sender/replay_cicids2018_labels.csv python
papers/soict2026/analyze_runs.py runs/run_split_20261019_092236.json --clock-json runs/clock_split_20261019_092236_*`.
