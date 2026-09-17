# ONNX 2026 — Paper: ONNX-EdgeIDS, a Framework-Agnostic ONNX/NumPy Serving Toolkit

**Venue:** TBD — draft, full systems paper (LNCS format), complementary in scope to a separate, shorter **Software Impacts** submission for the same software (see `notes/paper_pitch.md` and `submission_checklist.md` in the software repo below). Pick a venue and update this file (and `manuscript/main.tex`'s title page) before submission.
**Status:** draft manuscript compiles, 0 undefined citations; most empirical numbers are real, from the software's own on-hardware validation run (`submission_checklist.md`); the one number still genuinely pending is end-to-end throughput/latency over a **wired** inter-board link (Jetson #2 is currently on Wi-Fi).

## Research question

Earlier deployment work on this project's Jetson-based two-stage IDS pipeline found that an accurate classifier can still be served too slowly for an 8 GB edge board, if the serving stack is inherited unchanged from training. This paper presents the resulting software response: **ONNX-EdgeIDS**, a two-runtime (ONNX Runtime / pure NumPy) serving toolkit built around a single ONNX artifact, with a framework-agnostic NumPy exporter (reads standard `ai.onnx.ml.TreeEnsembleClassifier`/`Scaler` operator attributes directly, not any particular training framework's internal format) and a release-gated parity validator, integrated into a two-stage Kafka gate-then-classifier pipeline.

No PySpark/Apache Spark content: this paper (and the software it describes) is scoped entirely to serving an already-exported ONNX artifact — how that artifact was produced upstream is out of scope here.

## Title

*ONNX-EdgeIDS: A Framework-Agnostic ONNX/NumPy Serving Toolkit for Streaming Intrusion Detection on Jetson Orin Nano Super*

## Related code

The software this paper describes lives in its own repository, not in this monorepo:

| Component | Path (in [`vuthainguyen1602/onnx-edge-ids`](https://github.com/vuthainguyen1602/onnx-edge-ids)) |
|-----------|------|
| Swappable engine contract | `src/onnx_edge_ids/inference_engine.py` |
| Feature assembly | `src/onnx_edge_ids/feature_matrix.py` |
| ONNX backend | `src/onnx_edge_ids/onnx_engine.py` |
| NumPy backend | `src/onnx_edge_ids/numpy_engine.py` |
| Framework-agnostic ONNX → NumPy exporter | `scripts/export_numpy.py` |
| Parity release gate | `scripts/validate_parity.py` |
| Jetson #1 gate node | `scripts/gate_node.py` |
| Jetson #2 classifier node | `scripts/classifier_node.py` |
| CICIDS2017-derived replay producer | `scripts/csv_producer.py` |
| Two-Jetson deployment guide | `docs/deploy_2jetson.md` |
| Measurement/table plan | `experiments/plan.md` |
| Real validation-run numbers used in this manuscript | `submission_checklist.md` |
| Zenodo archive (concept DOI) | `10.5281/zenodo.22731927` |

Anomaly-gate artifacts (`anomaly_autoencoder.pkl`, `anomaly_scaler.pkl`, `anomaly_threshold.json`) are produced by the earlier thesis pipeline and copied in, not retrained by this toolkit.

## Reproduce

**Step 1 — export the NumPy fallback from an ONNX artifact, then validate parity (fills Table `tab:parity`):**

```bash
git clone https://github.com/vuthainguyen1602/onnx-edge-ids
cd onnx-edge-ids
ONNX_MODEL=artifacts/ids_rf.onnx FEATURES_JSON=artifacts/feature_columns.json ./scripts/export_artifacts.sh
python scripts/validate_parity.py --csv <path to a CICIDS2017-derived replay CSV> --rows 2000 \
  --report-json results/parity_report.json
```

**Step 2 — two-Jetson deployment, one classifier-node run per backend and per batch size (fills Tables `tab:engines`, batch-size sensitivity in Sect. "Results"):**

Follow `docs/deploy_2jetson.md` in the software repo: gate on Jetson #1, `classifier_node.py --engine onnx|numpy --batch-size 1|20|500 --metrics-csv ...` on Jetson #2.

**Step 3 — wired-link end-to-end run (the pending measurement):**

Repeat Step 2 over a wired inter-board link instead of Wi-Fi, and report producer-timestamp-to-verdict latency/throughput, not just in-process classifier cost.

## Metrics to report

- Classifier-node throughput (rows/s) and p95 (ms/row), per backend and per batch size
- Cold-start time to first verdict, per backend
- RSS (MB) and total-board power (W, `tegrastats`, not idle-subtracted)
- Label agreement / max confidence deviation between backends (parity gate)
- Artifact size (ONNX vs. NumPy)
- End-to-end (producer-to-verdict) throughput/latency over a **wired** link — pending

## Manuscript

The English draft is in `manuscript/` (Springer LNCS template, XeLaTeX). It compiles cleanly (`./manuscript/compile.sh`, 10 pages, 0 undefined citations/references). The numbers in Sect. "Results" are real (from the software's own validation run); the one remaining `\ph{...}` is the wired-link end-to-end measurement.

## Cross-reference

- Software artifact and short-form Software Impacts submission for the same toolkit: [`vuthainguyen1602/onnx-edge-ids`](https://github.com/vuthainguyen1602/onnx-edge-ids) (`notes/paper_pitch.md`, `submission_checklist.md`) — that submission stays scoped to the software itself; this LNCS draft is the fuller systems-paper treatment (architecture, related work, discussion) and the two are meant to be read independently, not merged.
- Prior distributed-deployment context (out of scope here): [`../soict2026/`](../soict2026/)
- Model selection (DT, SHAP Top-30): [`../fair2026/`](../fair2026/)
- Full thesis: [`../../thesis/`](../../thesis/)
