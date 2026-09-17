# Manuscript — ONNX 2026 (Springer LNCS)

English draft based on `Latex-Template-for-Springer/` (llncs.cls + splncs04.bst), same template as `../../soict2026/manuscript/`.

## Compile

```bash
cd papers/onnx2026/manuscript
./compile.sh
```

Compiles cleanly as of this draft (10 pages, 0 undefined citations/references — only cosmetic under/overfull-hbox warnings, same category the SOICT manuscript also has).

## TODO before submission

- [ ] Pick a venue and deadline, update `../README.md` and the title page
- [x] Author names and affiliations (copied from the SOICT companion — re-confirm order/roles before submission; note the software artifact itself, `onnx-edge-ids`, is single-authored by Thai Nguyen Vu per its `CITATION.cff` — confirm this paper's author list is what's intended)
- [x] No PySpark/Spark content anywhere in `main.tex`/`references.bib` (verified: `grep -in "pyspark\|spark"` returns nothing)
- [x] Most of Table `tab:engines`/`tab:parity`/`tab:coldstart` are real numbers from `onnx-edge-ids/submission_checklist.md`'s validation run — cross-check against a fresh run before submission, since the checklist itself flags these as a single run, not yet repeated
- [ ] Run the wired-link end-to-end benchmark and fill the one remaining `\ph{...}` (Sect. "Results"/"Discussion"/"Conclusion")
- [ ] Repeat the steady-state/cold-start/parity measurements across multiple runs (the checklist's own acceptance criteria call for this; the current numbers are a single run)
- [ ] Verify target venue's page limit/template requirements once chosen (the SOICT companion verified Springer CCIS/LNCS format for SOICT 2026; do not assume the same venue accepts this without checking)
- [ ] Self-contained submission package check (no `\input{../../../...}` outside the repo) once a venue is picked
- [ ] Decide whether/how to cross-reference the parallel Software Impacts submission for the same software once that one is finalized, so the two don't duplicate claims
