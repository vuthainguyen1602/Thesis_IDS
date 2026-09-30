# SOICT edge benchmark, re-measured (2026-09-29/30)

The published SOICT numbers (`../benchmarks/`, Table 4 of the manuscript) were
re-measured after checking the deployed system against the paper. This folder
holds the new measurements; the old files are left untouched.

## What was wrong with the deployed system

1. **Classifier features.** `jetson/model/feature_columns.json` held 29 features
   from a SHAP ranking made before the leakage fix; they shared 15 of 30 with the
   companion study's leakage-free SHAP Top-30. The export scripts now read the
   Top-30 from `results/ml_06_feature_selection_shap/shap_feature_importance.csv`.
   The retrained Random Forest (200 trees, depth 15, full training set) has test
   F1 **0.9959** (the companion study reports 0.9955).
2. **Gate features.** The autoencoder gate scored the classifier's feature list.
   On the new SHAP Top-30 it forwarded only 44.8% of attack flows at the
   pre-committed threshold (benign-validation quantile 0.995); on all 77
   leak-free features it forwards **68.1%** (FPR 0.49%, skip 89.4%). The gate now
   has its own list, `model/anomaly_feature_columns.json`.
3. **Batching.** Each tier processed a batch only once 20 flows had arrived, with
   no time limit, so the classifier (≈11% of traffic behind the gate) could hold
   a partial batch of attacks indefinitely. Batches now also flush after 1 s
   (`EDGE_BATCH_MAX_WAIT_S`).

## What was wrong with the measurement

- **Queued flows were not counted.** Throughput was the number of verdicts
  written inside the load window. The PySpark classifier cannot keep up, so its
  verdicts land after the window: the published 95.8% gate skip is
  1 − 2.6/62.5, the classifier's capacity divided by the total, not a property
  of the gate. The published p95 was per-node inference time and excluded
  queueing.
- **No gate in single/horizontal.** The `full` role enables the gate only with
  `ANOMALY_ENABLED=1`, which the orchestrator never set; those modes ran Spark on
  every flow. A `single_gate` mode was added.
- **Sender rate.** The replay loop reaches ≈80 flows/s when asked for 100.
- **Shared consumer group.** Flows a stopped run had not consumed were processed
  by the next run; each run now gets its own group.

`../../analyze_runs.py` follows every flow *sent* in a load window, using the
send timestamp and replay row now stored with each verdict (end-to-end latency
includes queueing; skip ratio and attack recall come from the flows themselves).

## Capacity (highest rate at which every flow got its verdict ≤10 s after load)

| Mode | Boards | Sustained | e2e p95 there | Live attack recall |
|---|---|---|---|---|
| Single node, gate off (as published) | 1 | ≈2.5 flows/s | 10–13 s | — |
| Horizontal, gate off (as published) | 2 | ≈3.3–4.8 | 8–13 s | — |
| Single node, gate on | 1 | ≈2–2.4 | 8–13 s | — |
| **A: pipeline split** | 2 | **≈22** | 5–9 s | **≈70%** |
| C: Spark cluster | 2 | ≈9–15 | 4–6 s (at 10) | ≈69% |

Source: `capacity_sweep_part1.csv` (rates run before 00:39) and
`capacity_sweep_part2.csv`. Runs between 00:39 and 08:04 were lost to the Mac
sleeping and were repeated (the two single_gate @ 2 rows from that window were
dropped from part 1: 65 s load window, sender at 1.3 flows/s); `capacity_sweep_before_batch_flush.csv` is the sweep
before the batching fix, kept as evidence of the stall.

## Energy at the sustained rate (`energy_at_sustained_rate.csv`)

| Mode | Rate | Power | J / verdict |
|---|---|---|---|
| Single node | 2.0/s | 6.1 W | 3.04 |
| Horizontal | 3.9/s | 12.5 W | 3.20 |
| Single node, gate on | 2.0/s | 6.1 W | 3.04 |
| **A: pipeline split** | 18.4/s | 11.0 W | **0.60** |
| C: Spark cluster | 9.5/s | 10.9 W | 1.15 |

## Same forest, three engines (`benchmark_same_forest*.json`, Jetson #2, batch 10)

| Engine | Flows/s | Batch p95 | Active energy / flow |
|---|---|---|---|
| PySpark (deployed) | 1.4 | 7.5 s | 1506 mJ |
| NumPy | 54.8 | 194 ms | 11.3 mJ |
| ONNX Runtime | 10,092 | 1.2 ms | 0.087 mJ |

The three engines agree on 100% of predictions. `benchmark_engines_sklearn_retrained.json`
is the old-style comparison (a separately trained scikit-learn forest) and is
not the same model.

## Reproduce

```bash
cd papers/soict2026
./run_dist_bench.sh split                     # one mode, with energy window
MODES="split" RATES_split="10 20 30" ./run_capacity_sweep.sh
python analyze_runs.py results/benchmarks/run_split_rep*_<tag>.json
# on Jetson #2
python scripts/benchmark_same_forest.py --samples 1000 --batch-size 10
```

Keep the Mac awake (`caffeinate -i`, lid open or on power) during a sweep.
