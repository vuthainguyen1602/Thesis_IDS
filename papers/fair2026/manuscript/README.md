# Manuscript — FAIR'2026 (IEEE conference template)

Built on the IEEE conference template (`IEEEtran.cls` + `ieeetr.bst`) in `../IEEE-conference-template-062824/`, compiled with **XeLaTeX** for Vietnamese.

## Files

| File | Description |
|------|-------------|
| `main.tex` | The paper (Vietnamese, IEEE 2-column, ~6 pages) |
| `references.bib` | BibTeX references |
| `compile.sh` | Build the PDF (xelatex → bibtex → xelatex ×2) |

## Build

```bash
# Collect figures first (from the repo root)
./papers/fair2026/collect_results.sh

# Compile
cd papers/fair2026/manuscript
./compile.sh
# → main.pdf
```

**Requirements:** TeX Live with `xelatex` and `bibtex`; packages `IEEEtran`, `fontspec`, `polyglossia`, `booktabs`, `amsmath`, `multirow`. The Vietnamese main font is set in `../../latex/fonts-xelatex-vi.tex` (Times New Roman by default).

```bash
# macOS — full TeX Live if packages are missing
# brew install --cask mactex-no-gui
```

## TODO before submission

- [x] Author names, affiliations, emails
- [x] Related work
- [ ] Fill the `\ph{...}` placeholders with real values after running the pipeline on the Jetson cluster
- [ ] Final format check against the [FAIR'2026](https://fair.conf.vn/) instructions
