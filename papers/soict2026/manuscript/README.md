# Manuscript — SOICT 2026 (Springer LNCS)

English paper based on `Latex-Template-for-Springer/` (llncs.cls + splncs04.bst).

## Compile

```bash
./papers/soict2026/collect_results.sh
cd papers/soict2026/manuscript
./compile.sh   # pdfLaTeX (Springer production); Times via mathptmx, Type 1 fonts only
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

## Camera-ready (accepted, paper 3215)

- Limit: **12–15 pages including references** (CCIS full paper); the PDF is 15.
- Fig. 5 and Fig. 6 are vector PDFs drawn at print size (lettering ≥ 6 pt):
  `python papers/soict2026/plot_capacity.py` writes `edge_capacity.pdf`,
  `python ml_08_anomaly_gate_autoencoder.py --replot` writes `gate_operating_points.pdf`;
  `compile.sh` copies both into `figures/`. The PNGs stay for the decks and the thesis.
- Checked against Springer's "Instructions for Authors of Papers to be Published in Springer
  Computer Science Proceedings": captions that are not sentences end without a period, multiple
  citations sorted (`cite`), corresponding author marked (envelope + email), ORCID of the first author.
- Authors as in EasyChair: Thai Nguyen Vu, Bui Van Dung, Tri Nhut Do, Van Du Nguyen
  (family names Vu, Bui, Do, Nguyen); running head "T. N. Vu et al.".
- Upload: PDF + ZIP (`Source/`, corresponding-author txt, ink-signed scanned copyright form).
