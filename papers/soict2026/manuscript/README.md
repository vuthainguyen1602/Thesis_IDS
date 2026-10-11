# Manuscript — SOICT 2026 (Springer LNCS)

English paper based on `Latex-Template-for-Springer/` (llncs.cls + splncs04.bst).

## Compile

```bash
./papers/soict2026/collect_results.sh
cd papers/soict2026/manuscript
./compile.sh
```

## TODO before submission

- [x] Author names and affiliations
- [x] Architecture figure — drawn inline with TikZ (Fig. 1), no `architecture.pdf` needed
- [x] Table~\ref{tab:benchmark} filled from the Jetson benchmark JSONs in `../results/benchmarks/`
- [x] Flatten the submission package: `llncs.cls`, `splncs04.bst`, `references.bib` and `figures/` live in this folder; `compile.sh` refreshes the figures from their sources when present
- [x] Verify SOICT page limit and formatting at [soict.org](https://soict.org/) — **verified 2026-07-02**:
  - SOICT 2026 proceedings: **Springer CCIS series**; papers must follow **LNCS/CCIS format** → current `llncs.cls` + `splncs04.bst` is correct.
  - Page limit: **max 12 pages excluding references**; PDF, **no page numbers**; single-blind (keep author names); language: English.
  - Submission via EasyChair; Abstract 09/09/2026, Full paper **16/09/2026**, Notification 12/10/2026, Camera-ready 23/10/2026; conference 4–5/12/2026, TP.HCM.
