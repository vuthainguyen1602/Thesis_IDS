# SOICT 2026 — Paper: Distributed Edge IDS on Jetson Orin Nano Super

**Venue:** [SOICT 2026 — International Symposium on Information and Communication Technology](https://soict.org/)
**Abstract deadline:** 2026-09-09 | **Full paper:** 2026-09-16 | **Conference:** 2026-12-04/05

## Research question

Architecture and performance evaluation of a real-time distributed IDS on 2× NVIDIA Jetson Orin Nano Super using Kafka and PySpark.

## Title

*A Distributed Real-Time Intrusion Detection System on a Jetson Orin Nano Super Edge Cluster using Kafka and PySpark*

## Related code (repo root)

| Component | Path |
|-----------|------|
| Edge pipeline | `jetson/edge/kafka_consumer.py` |
| Distributed roles | `jetson/edge/role_pipelines.py` |
| Jetson guide | `jetson/JETSON_DISTRIBUTED.md` |
| Anomaly gate | `ml_08_anomaly_gate_autoencoder.py` |
| Model export | `jetson/scripts/save_model.py` |
| Benchmark | `jetson/scripts/benchmark.py` |
| Env templates | `jetson/.env.jetson1.example`, `.env.jetson2.example` |

## Reproduce

**Step 1 — Mac (train + export + infra), dispatched to the Jetson cluster:**

```bash
export IDS_ROOT="$(pwd)"
./cluster/reproduce_cluster.sh soict
```

**Step 2 — 2× Jetson Orin Nano Super (edge benchmark):**

```bash
./papers/soict2026/run_benchmarks.sh   # local | run | merge → summary.csv
```

**Step 3 — Collect results:**

```bash
./papers/soict2026/collect_results.sh
python papers/soict2026/plot_edge_modes.py   # → edge_modes.png
```

## Three distributed modes (compared in the paper)

| Mode | Jetson #1 | Jetson #2 | Env |
|------|-----------|-----------|-----|
| A — Pipeline split | `EDGE_NODE_ROLE=anomaly_gate` | `EDGE_NODE_ROLE=classifier` | `.env.jetson1/2.example` |
| B — Horizontal | `EDGE_NODE_ROLE=full` | `EDGE_NODE_ROLE=full` | `.env.jetson-horizontal.example` |
| C — Spark cluster | Spark worker | Spark worker | `cluster/start_worker.sh` (Mac = master) |

## Metrics to report

- Throughput (flows/s)
- Latency p50 / p95 (ms)
- CPU %, RAM (MB), temperature (°C) per node
- Attack-class F1
- Gate-skip ratio — % of flows filtered by the anomaly gate (mode A)
- Energy per inference (mJ) via `tegrastats` (`jetson/edge/power_monitor.py`)

Benchmark CSV/JSON is written to `papers/soict2026/results/benchmarks/` before `collect_results.sh`.

> **Published numbers vs. the current tree.** The paper's latency/throughput were
> measured before `jetson/edge/role_pipelines.py` gained the swappable
> `EDGE_ENGINE` backend. Back then the per-batch timer wrapped `model.transform`
> only — lazy in Spark, with the `count()` that forced the job outside the timer —
> whereas `edge/inference_engine.py` now times DataFrame construction through
> `collect()`, i.e. real execution. Feature values are unaffected (the Spark path
> stays float64 via `dtype_for_engine`). So re-running Modes A/B/C from this tree
> reproduces the *architecture* and the throughput comparison, but its latency
> percentiles sit above the published ones by construction; do not mix the two in
> one table.
>
> The gate-skip denominator was also corrected afterwards: it now counts every
> final verdict in the load window, whereas the published runs counted gate skips
> plus attack verdicts only and so dropped the classifier's benign verdicts on
> forwarded flows. The effect is small — on `run_split_rep0_…153003.json` the
> stored 95.6 becomes 95.47 — but a re-run will land a fraction of a point below
> the paper's 95.8 / 97.9.

## Manuscript

The English paper is in `manuscript/` (Springer LNCS template, XeLaTeX).

## Cross-reference

- Model selection (DT, SHAP Top-30): see the FAIR paper / `papers/fair2026/`
- Full thesis: `thesis/`
