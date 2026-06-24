#!/bin/bash
# Compile the merged journal draft (LNCS + XeLaTeX).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
SPRINGER="$ROOT/Latex-Template-for-Springer"
BIB="$ROOT/papers/latex"
OUT="$ROOT/output/pdfs"
export TEXINPUTS="${SPRINGER}//:${TEXINPUTS:-}"
export BSTINPUTS="${BIB}//:${SPRINGER}//:${BSTINPUTS:-}"
cd "$DIR"; mkdir -p "$OUT"
xelatex -interaction=nonstopmode main.tex || true
bibtex main || true
xelatex -interaction=nonstopmode main.tex || true
xelatex -interaction=nonstopmode main.tex || true
cp main.pdf "$OUT/JOURNAL2026_draft.pdf"
echo "PDF: $DIR/main.pdf and $OUT/JOURNAL2026_draft.pdf"
