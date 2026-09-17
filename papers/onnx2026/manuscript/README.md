# Manuscript — ONNX 2026 (Springer LNCS)

English draft based on `Latex-Template-for-Springer/` (llncs.cls + splncs04.bst), same template as `../../soict2026/manuscript/`.

## Compile

```bash
cd papers/onnx2026/manuscript
./compile.sh
```

Compiles cleanly as of this draft (11 pages, 0 undefined citations/references — only cosmetic under/overfull-hbox warnings, same category the SOICT manuscript also has).

## TODO before submission

- [ ] Pick a venue and deadline, update `../README.md` and the title page
- [x] Author names and affiliations (copied from the SOICT companion — re-confirm order/roles before submission)
- [ ] Run `scripts/export_onnx.py --validate-csv` and fill the Spark-vs-ONNX / Spark-vs-NumPy agreement rate (Sect. "Conversion Correctness")
- [ ] Benchmark `EDGE_ENGINE=spark|numpy|onnx` end-to-end on the 2× Jetson cluster and fill Table `tab:pipeline-engines` + Fig. `engine-modes` (see `../README.md` reproduce steps)
- [ ] `plot_edge_modes.py`-style script for `engine_modes.png` once `results/benchmarks/` has data (adapt from `../../soict2026/plot_edge_modes.py`)
- [ ] Verify target venue's page limit/template requirements once chosen (the SOICT companion verified Springer CCIS/LNCS format for SOICT 2026; do not assume the same venue accepts this without checking)
- [ ] Self-contained submission package check (no `\input{../../../...}` outside the repo) once a venue is picked
