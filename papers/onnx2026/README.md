# ONNX 2026 — Paper: JVM-Free ONNX/NumPy Edge Serving for the IDS

**Venue:** TBD — draft written as a companion/follow-up to [`../soict2026/`](../soict2026/); pick a venue and update this file (and `manuscript/main.tex`'s title page) before submission.
**Status:** draft manuscript compiles; empirical results are placeholders pending an end-to-end benchmark on real Jetson hardware (see below).

## Research question

The SOICT companion paper measured that serving the deployed Random Forest through a JVM-hosted PySpark `PipelineModel` on an 8 GB Jetson Orin Nano Super is $9\times$–$348\times$ slower than the same forest under scikit-learn or ONNX Runtime (single-node microbenchmark), and left "wiring an ONNX session into the streaming pipeline" as future work. This paper closes that gap: a swappable classifier-tier contract (`spark | numpy | onnx`) exported directly from Spark's saved Parquet node tables (no scikit-learn or retraining step), integrated into the existing Kafka role pipeline behind one `EDGE_ENGINE` switch.

## Title

*From PySpark to ONNX: A Bit-Exact, JVM-Free Serving Path for an Edge Intrusion Detection System on Jetson Orin Nano Super*

## Related code (repo root, branch `feature/numpy-prediction-engine`)

| Component | Path |
|-----------|------|
| Swappable engine contract | `jetson/edge/inference_engine.py` |
| Engine-agnostic feature assembly | `jetson/edge/feature_matrix.py` |
| Spark backend (adapter, unchanged serving path) | `jetson/edge/spark_engine.py` |
| ONNX backend | `jetson/edge/onnx_engine.py` |
| NumPy backend | `jetson/edge/numpy_engine.py` |
| Spark → ONNX exporter | `jetson/scripts/export_onnx.py` |
| Spark → NumPy exporter | `jetson/scripts/export_numpy.py` |
| Export both artifacts | `jetson/scripts/export_edge_artifacts.sh` |
| Role-pipeline integration | `jetson/edge/role_pipelines.py` |
| Config (`EDGE_ENGINE`, `ONNX_*`, `NUMPY_*`) | `jetson/config.py` |
| ONNX Runtime dependency | `jetson/requirements_onnx.txt` |

## Reproduce

**Step 1 — export serving artifacts from the deployed Spark `PipelineModel`:**

```bash
cd jetson
./scripts/export_edge_artifacts.sh
#   → model/ids_rf.onnx
#   → model/ids_rf_numpy.npz
```

**Step 2 — validate conversion correctness (fills Sect. "Conversion Correctness" in the manuscript):**

```bash
python scripts/export_onnx.py \
  --model model/ids_pipeline_random_forest \
  --out   model/ids_rf.onnx \
  --validate-csv <path to a held-out CICIDS2017 sample>
```

**Step 3 — 2× Jetson Orin Nano Super, one repetition set per backend (fills Table `tab:pipeline-engines`):**

```bash
# On Jetson #2 (classifier role), repeat for EDGE_ENGINE=spark, numpy, onnx,
# gate on Jetson #1 held fixed — same protocol as papers/soict2026/run_benchmarks.sh.
EDGE_ENGINE=onnx ./papers/soict2026/run_benchmarks.sh   # or numpy / spark
```

**Step 4 — collect results:**

```bash
./papers/soict2026/collect_results.sh   # reuse the same collector, output to results/benchmarks/
```

## Metrics to report

- Throughput (flows/s), per `EDGE_ENGINE` backend
- Latency p50 / p95 (ms)
- CPU %, RAM (MB), temperature (°C) on the classifier node
- Energy per verdict (mJ) via `tegrastats`
- Spark-vs-export prediction agreement rate (label match + max probability deviation), offline and spot-checked online

Benchmark CSV/JSON should be written to `papers/onnx2026/results/benchmarks/` before drafting the results section further.

## Manuscript

The English draft is in `manuscript/` (Springer LNCS template, XeLaTeX; same template as `../soict2026/manuscript/`). It compiles cleanly (`./manuscript/compile.sh`) with all citations resolved; the numbers still marked `[TBD]`/`\ph{...}` need the benchmark run above before this is submission-ready.

## Cross-reference

- Deployed architecture, anomaly gate, and the single-node engine microbenchmark this paper builds on: [`../soict2026/`](../soict2026/)
- Model selection (DT, SHAP Top-30): [`../fair2026/`](../fair2026/)
- Full thesis: [`../../thesis/`](../../thesis/)
