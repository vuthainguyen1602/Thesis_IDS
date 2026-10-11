# Batch-size sweep, PySpark vs ONNX Runtime (Jetson #2, 2026-10-11)

Same forest as `../remeasure_20260930/benchmark_same_forest*.json` (the deployed
RF, 200 trees, depth 15, SHAP Top-30), served through PySpark and ONNX Runtime at
batch sizes 10 to 10,000. Question: is PySpark slow on the board because of the
trees, or because of a fixed cost per call?

Run with `jetson/scripts/sweep_batch_size.sh` (TAG=20261011) on Jetson #2,
idle otherwise. Spark and ONNX agree on 100% of predictions at every batch size.

| Batch | Spark flows/s | Spark p50 / batch | Spark mJ / flow | ONNX flows/s | ONNX mJ / flow |
|---|---|---|---|---|---|
| 10 | 1.4 | 7.09 s | 1,429 | 10,160 | 0.081 |
| 100 | 14.2 | 7.14 s | 144 | 25,987 | 0.107 |
| 1,000 | 134.5 | 7.43 s | 15.6 | 27,281 | 0.065 |
| 10,000 | 1,046 | 9.50 s | 2.0 | 28,589 | 0.039 |

Source of each row: Spark from `batch_<n>.json`; ONNX from
`batch_<n>_onnx_energy.json` (rows tiled so each run lasts 26-44 s). The ONNX
rows inside `batch_<n>.json` ran for 0.1-1.8 s, too short for tegrastats at 200 ms,
and serve only the agreement check. `batch_sweep_table.csv` is what
`plot_batch_sweep.py` plots (`batch_sweep.png`).

## Reading

- **Spark pays about 7 s per call.** From batch 10 to 1,000 the time per batch
  moves from 7.1 to 7.4 s, so throughput and energy per flow scale almost exactly
  with batch size. Only at 10,000 flows does the work itself show (+2.4 s).
  Spark's active power stays at 2.0-2.1 W throughout.
- **The ONNX/Spark ratio depends on batch size**: about 7,000x at batch 10 (the
  deployed setting, and the "~7,200x" in the paper), about 27x at batch 10,000.
  Quote the ratio with its batch size.
- A streaming IDS cannot batch its way out: at 20 flows/s, 10,000 flows take
  over 8 minutes to collect.

## Caveats

- One run per point, no repeats.
- ONNX energy is not monotonic in batch size. The idle baseline is stable
  (4.60-4.71 W), but active power differs between runs (0.82, 2.77, 1.77, 1.12 W
  for batch 10, 100, 1,000, 10,000). The cause is untested (thermal/DVFS state,
  ONNX thread use). Claim only the order of magnitude (0.04-0.11 mJ/flow), not a
  trend.
- Throughput includes building the feature matrix from Python dicts, as in the
  edge consumer.

Logs: `sweep_batch_20261011.trimmed.log` (Spark progress bars removed),
`onnx_energy_20261011.log`.
