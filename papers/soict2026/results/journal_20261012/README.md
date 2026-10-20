# Edge placements for the journal version (2026-10-12/13)

The conference table (`../remeasure_20260930/`) compares one board, horizontal
scaling, the two-board pipeline split and the Spark cluster, all with the
PySpark classifier and the gate off except in the split. Two placements were
missing (gate and classifier as *two processes on one board*; horizontal
scaling *with the gate on*), and the journal asks a new question: once the
classifier runs on ONNX Runtime instead of PySpark, is the anomaly gate still
worth the recall it costs? This folder holds those runs. Same boards, models,
replay file and Wi-Fi links as on 2026-09-30; same queue-aware metrics
(`../../analyze_runs.py`).

## Modes

| Mode | Boards | Jetson #1 | Jetson #2 | Classifier |
|---|---|---|---|---|
| split_1board | 1 | – | gate process + classifier process | PySpark |
| horizontal_gate | 2 | full, gate on | full, gate on | PySpark |
| single_onnx | 1 | – | full, gate off | ONNX Runtime |
| single_gate_onnx | 1 | – | full, gate on | ONNX Runtime |
| split_1board_onnx | 1 | – | gate process + classifier process | ONNX Runtime |
| split_onnx | 2 | gate | classifier | ONNX Runtime |
| horizontal_onnx | 2 | full, gate off | full, gate off | ONNX Runtime |
| horizontal_gate_onnx | 2 | full, gate on | full, gate on | ONNX Runtime |

"Full" is one process doing gate (if on) and classifier; `EDGE_BATCH_SIZE=20`,
`EDGE_BATCH_MAX_WAIT_S=1`, Postgres and InfluxDB on the Mac, one autocommit
INSERT per verdict, as deployed. ONNX modes use `EDGE_ENGINE=onnx` on the
same 200-tree forest (`jetson/model/ids_rf.onnx`); the gate is the same
autoencoder on all 77 leak-free features.

## Procedure

`../../run_journal_sweep.sh` (log: `sweep.log`) ran the eight modes in the
order above. Per mode: a capacity sweep (`run_capacity_sweep.sh`, two repeats
per offered rate, 15 s warm-up, 45 s load, stop at the first rate whose worst
repeat drained more than 10 s after the load window), then one run with the
`tegrastats` energy window at the highest rate that was sustained
(`run_dist_bench.sh <mode>`, 45 s load after a 30 s idle baseline). Points
whose load window straddled a step of the Mac's clock (below) were repeated
afterwards (`rerun.log`, `rerun2.log`, `rerun3.log`).

**Sender.** The replay loop was rewritten to deadline pacing (`_timestamp`
of row *i* = start + *i*/rate), so the offered rate is met up to about
1,000 flows/s (the old loop topped out near 80). Rows wrap around the
10,000-row file when a window needs more; `_row` still indexes the labels.

**Clocks.** End-to-end latency is verdict time (board clock) minus send time
(Mac clock). Both boards are NTP-synchronised to each other within 2 ms; the
Mac was 0.67 s behind them at the start and, having no passwordless sudo for
`sntp -sS`, was left alone. Every run therefore records the board-minus-Mac
offset right before and right after its load window
(`jetson/scripts/clock_offset.py`, ±2 ms over one SSH channel; for the first
three modes only the orchestrator's per-mode `clock_<mode>_<T>_{before,after}.json`
exist) and `analyze_runs.py` subtracts it. During the night macOS stepped
its clock several times by 0.08-0.48 s (0.67 → 0.20 → 0.37 → 0.08 s against
NTP; `mac_clock_sntp.txt`, `clock_start.json`, `clock_end.json`). A repeat
whose two snapshots differ by more than 50 ms is marked `tainted` in
`capacity_all.csv` and excluded from the latency means in
`capacity_summary.csv`; throughput, drain, skip and recall do not depend on
the clock beyond ±0.5 s on drain, so the saturation rule uses every repeat.
Seven of 72 repeats are tainted; every (mode, rate) point keeps at least two
clean repeats.

**Jetson #1** also hosts another project's k3s cluster (ArgoCD, a Spring
application, Postgres) and a Playwright container, together about 50% CPU
and 5 GB RSS. k3s was stopped (`k3s-killall.sh`) for the measurements and
started again afterwards (`boards_before.txt`: 677 MB used, load 1.2 before
the first mode). Both boards in 25 W mode; Wi-Fi link 175-390 Mb/s;
Mac → board ping 3.6-9.5 ms average per snapshot, median 4.9 ms over the PySpark runs (`ping_rtt.csv`, from every clock snapshot), board → board 6.3 ms average
(`rtt_j1_to_j2.txt`). onnxruntime 1.23.2 was installed into Jetson #1's
venv for the horizontal ONNX modes (Jetson #2 already had it).

## Capacity (`capacity_summary.csv`; every repeat in `capacity_all.csv`)

Sent/s is what the sender achieved; sustained/s = completed verdicts over
(last verdict − load start); drain = last verdict − load end, worst of the
repeats; latencies are means over the clean repeats; recall = attack flows
given an Attack verdict / attack flows sent.

| Mode | Boards | Offered | Sent/s | Sustained/s | Drain max (s) | e2e p50 / p95 (ms) | Skip % | Recall % | Clean reps |
|---|---|---|---|---|---|---|---|---|---|
| split_1board | 1 | 5 | 4.9 | 4.6 | 5.5 | 646 / 4772 | 89.8 | 65.6 | 2/2 |
| split_1board | 1 | 10 | 9.6 | 8.6 | 6.1 | 618 / 5652 | 89.1 | 70.8 | 2/2 |
| split_1board | 1 | 20 | 19.6 | 17.8 | 4.5 | 565 / 5252 | 88.9 | 67.6 | 2/2 |
| split_1board | 1 | 30 (saturated) | 29.0 | 21.7 | 16.1 | 383 / 11773 | 89.0 | 68.8 | 2/2 |
| horizontal_gate | 2 | 3 | 3.0 | 2.9 | 1.8 | 4077 / 7969 | 88.2 | 77.8 | 2/2 |
| horizontal_gate | 2 | 5 | 4.9 | 4.4 | 8.3 | 4555 / 8624 | 89.8 | 65.6 | 2/2 |
| horizontal_gate | 2 | 8 (saturated) | 7.9 | 4.6 | 32.6 | 15924 / 28510 | 88.6 | 70.4 | 2/2 |
| single_onnx | 1 | 50 | 48.5 | 47.7 | 0.9 | 212 / 393 | 0.0 | 100.0 | 2/2 |
| single_onnx | 1 | 100 | 94.8 | 92.7 | 1.0 | 153 / 961 | 0.0 | 99.7 | 2/2 |
| single_onnx | 1 | 200 (saturated) | 195.4 | 188.1 | 10.3 | 284 / 939 | 0.0 | 99.8 | 2/4 |
| single_gate_onnx | 1 | 50 | 49.0 | 48.1 | 0.9 | 212 / 393 | 88.8 | 69.6 | 2/2 |
| single_gate_onnx | 1 | 100 | 98.2 | 96.0 | 1.1 | 107 / 197 | 89.1 | 68.7 | 2/2 |
| single_gate_onnx | 1 | 200 | 196.2 | 172.1 | 8.7 | 2869 / 5523 | 89.0 | 69.3 | 2/2 |
| single_gate_onnx | 1 | 300 (saturated) | 293.1 | 161.0 | 44.9 | 20919 / 34846 | 88.9 | 69.7 | 2/2 |
| split_1board_onnx | 1 | 50 | 49.0 | 46.7 | 2.7 | 246 / 1116 | 88.8 | 69.6 | 2/2 |
| split_1board_onnx | 1 | 100 | 97.1 | 93.0 | 2.1 | 139 / 900 | 89.1 | 68.7 | 2/2 |
| split_1board_onnx | 1 | 200 (saturated) | 196.2 | 166.9 | 30.1 | 5469 / 10640 | 89.0 | 69.3 | 3/4 |
| split_onnx | 2 | 50 | 49.1 | 47.2 | 1.9 | 236 / 1078 | 88.8 | 69.6 | 2/2 |
| split_onnx | 2 | 100 | 98.2 | 93.9 | 2.1 | 121 / 789 | 89.1 | 68.7 | 2/2 |
| split_onnx | 2 | 200 | 195.7 | 181.1 | 4.0 | 1625 / 2989 | 89.0 | 69.3 | 2/2 |
| split_onnx | 2 | 300 (saturated) | 293.1 | 179.3 | 29.7 | 15370 / 26696 | 88.9 | 69.7 | 2/2 |
| horizontal_onnx | 2 | 50 | 49.0 | 48.2 | 0.9 | 402 / 824 | 0.0 | 100.0 | 2/2 |
| horizontal_onnx | 2 | 100 | 98.0 | 96.3 | 0.9 | 206 / 419 | 0.0 | 99.7 | 2/2 |
| horizontal_onnx | 2 | 200 | 195.8 | 191.7 | 1.0 | 108 / 213 | 0.0 | 99.8 | 2/2 |
| horizontal_onnx | 2 | 300 | 294.5 | 287.8 | 4.0 | 494 / 1142 | 0.0 | 99.8 | 3/4 |
| horizontal_onnx | 2 | 500 (saturated) | 491.1 | 258.7 | 46.0 | 29067 / 39784 | 0.0 | 99.8 | 2/2 |
| horizontal_gate_onnx | 2 | 50 | 49.1 | 48.1 | 1.1 | 407 / 838 | 88.8 | 69.6 | 3/4 |
| horizontal_gate_onnx | 2 | 100 | 98.2 | 96.3 | 0.9 | 219 / 519 | 89.1 | 68.7 | 2/2 |
| horizontal_gate_onnx | 2 | 200 | 196.1 | 191.8 | 1.0 | 122 / 527 | 89.0 | 69.3 | 2/2 |
| horizontal_gate_onnx | 2 | 300 | 294.8 | 288.4 | 1.1 | 111 / 473 | 88.9 | 69.7 | 2/4 |
| horizontal_gate_onnx | 2 | 500 (saturated) | 488.1 | 223.0 | 48.5 | 32119 / 49426 | 88.9 | 69.5 | 2/2 |

Read with the 2026-09-30 table (single node 2.5 flows/s, horizontal 3.3,
two-board split ≈22, Spark cluster ≈9-17):

- **Placement, PySpark.** The pipeline split needs no second board: both
  tiers as two processes on Jetson #2 sustain 20 flows/s (two boards: ≈22).
  Horizontal scaling with the gate on reaches 5 flows/s, twice the gate-off
  figure (3.3) but a quarter of the split; the gate inside the Spark process
  only helps as much as Spark is relieved, and both boards still carry a JVM.
- **ONNX moves the bottleneck.** With the classifier at ~15,000 rows/s the
  pipeline is bound by the verdict path: one Kafka consume, one feature
  matrix and one Postgres INSERT per flow over Wi-Fi. One board without the
  gate sustains 100 flows/s (200 is marginal: 10.3 s drain, 188 verdicts/s);
  with the gate 200 (300 saturates). Two boards: split 200, horizontal
  300 with or without the gate.
- **Is the gate still worth it?** With PySpark the gate was the difference
  between 2.5 and 22 flows/s (9×). With ONNX it buys at most 2× on one board
  (100 → 200) and nothing on two boards in horizontal mode (300 → 300),
  while it still drops live attack recall from ≈100% to ≈69% at a skip ratio
  of 89%. Latency at the common operating points is similar with and without
  it (e.g. 100 flows/s on one board: p95 0.96 s without, 0.20 s with; 200
  flows/s on two boards horizontal: 0.21 s without, 0.53 s with).
- **Recall is a property of the gate, not the placement.** Every gated mode
  gives 69-70% at 100-300 flows/s (65-78% at the lowest rates, where the
  window holds only 30-140 attack flows).

## Energy at the sustained rate (`energy_at_sustained_rate.csv`; all energy runs in `energy_all.csv`)

Whole-board average power over the 45 s load (sum over the participating
boards), divided by the flows sent per second; the active column subtracts
the 30 s idle baseline taken just before. The latest clean energy run per
mode is listed; the rate is the highest the sweep sustained, except where
that run itself queued (see `drain_s`), in which case a lower rate was run.

| Mode | Boards | Rate (sent/s) | Drain (s) | Board power (W) | Active (W) | J / verdict | Active mJ / verdict | e2e p50 / p95 (ms) | Skip % | Recall % |
|---|---|---|---|---|---|---|---|---|---|---|
| split_1board | 1 | 20 (19.4) | 4.4 | 6.15 | 1.52 | 0.317 | 78.4 | 564 / 5409 | 88.9 | 67.6 |
| horizontal_gate | 2 | 5 (4.9) | 5.5 | 12.1 | 3.01 | 2.469 | 614.3 | 4327 / 7924 | 89.8 | 65.6 |
| single_onnx | 1 | 100 (98.2) | 1.1 | 4.63 | 0.03 | 0.047 | 0.3 | 109 / 202 | 0.0 | 99.7 |
| single_gate_onnx | 1 | 100 (98.2) | 1.1 | 4.66 | 0.07 | 0.047 | 0.7 | 116 / 209 | 89.1 | 68.7 |
| split_1board_onnx | 1 | 100 (98.3) | 2.1 | 4.68 | 0.05 | 0.048 | 0.5 | 120 / 796 | 89.1 | 68.7 |
| split_onnx | 2 | 200 (194.8) | 5.1 | 9.28 | 0.14 | 0.048 | 0.7 | 2396 / 4510 | 89.0 | 69.3 |
| horizontal_onnx | 2 | 200 (196.3) | 1.0 | 9.62 | 0.3 | 0.049 | 1.5 | 108 / 213 | 0.0 | 99.8 |
| horizontal_gate_onnx | 2 | 200 (196.5) | 1.0 | 9.55 | 0.06 | 0.049 | 0.3 | 107 / 212 | 89.0 | 69.3 |

Against 2026-09-30 (single node 3.04 J/verdict at 2.0 flows/s, horizontal
3.20 at 3.9, two-board split 0.60 at 18.4, Spark cluster 1.15 at 9.5):

- **One board does the split at half the energy of two**: 0.32 J/verdict at
  19.4 flows/s on Jetson #2 alone against 0.60 on two boards, because the
  second board's 5 W idle is gone and the gate costs the classifier board
  almost nothing. Horizontal with the gate on: 2.47 J/verdict at 4.9
  flows/s, two JVMs for a fifth of the split's throughput.
- **With ONNX the board is idle.** Whole-board power is 4.6-4.7 W on one
  board and 9.3-9.6 W on two, within 0.1-0.6 W of the idle baseline at
  100-200 flows/s, so energy per verdict is idle power divided by rate:
  0.047-0.049 J at every placement, 12x below the PySpark split and 65x
  below single-node PySpark. Placement and gate change the energy figure
  only through the rate they let the board run at.
- The 300 flows/s energy runs of the two horizontal ONNX modes and the 200
  flows/s run of single_gate_onnx queued (drain 8.5-19.6 s, p50 1.7-16 s)
  although the sweep repeats at those rates had not; they stay in
  `energy_all.csv`, the table uses the re-runs at 200 and 100 flows/s.


## Files

- `runs/run_<mode>_rep<k>_*.json` – sweep repeats (driver output, incl. the
  clock snapshots from mode 4 onwards); `runs/run_<mode>_<T>.json` – the
  energy runs; `runs/power_<node>_<T>.json` – tegrastats windows;
  `runs/clock_<mode>_<T>_{before,after}.json` – per-mode clock/RTT snapshots;
  `runs/ids_raw_latency_*.csv` – per-flow inference logs pulled from the
  boards (42 MB, cumulative since September, not committed); `runs/capacity_sweep*.csv`, `runs/energy_runs.csv` – the
  analyzer's rows as written during the night (superseded by the tables
  above, which `../../journal_tables.py` re-derives from the JSONs with the
  per-run clock offsets).
- `sweep.log`, `rerun*.log` – full orchestrator output; `boards_*.txt`,
  `rtt_j1_to_j2.txt`, `mac_clock_sntp.txt`, `clock_{start,end}.json`.

Reproduce: `../../run_journal_sweep.sh` (Mac, Docker services up, both
boards reachable, Mac kept awake), then
`python ../../journal_tables.py runs`.
