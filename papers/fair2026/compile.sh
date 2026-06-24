#!/bin/bash
# Build FAIR'2026 (IEEE conference format, XeLaTeX + Vietnamese).
# Requires: IEEEtran.cls (bundled in ./IEEE-conference-template-062824/),
#           ieeetr.bst (ships with TeX Live), Times New Roman (macOS) or any
#           Vietnamese-capable serif set in ../latex/fonts-xelatex-vi.tex.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/manuscript"

export TEXINPUTS="$HERE/IEEE-conference-template-062824//:$HERE/../latex//:"
export BSTINPUTS="$HERE/../latex//:"
export BIBINPUTS="$HERE/../latex//:"

xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex

echo "Done -> $HERE/manuscript/main.pdf"
