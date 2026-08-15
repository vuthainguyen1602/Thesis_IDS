#!/bin/bash
# Compile SOICT 2026 paper (Springer LNCS + XeLaTeX)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
SPRINGER="$ROOT/Latex-Template-for-Springer"
BIB="$ROOT/papers/latex"
OUT="$ROOT/output/pdfs"

export TEXINPUTS="${SPRINGER}//:${TEXINPUTS:-}"
export BSTINPUTS="${BIB}//:${SPRINGER}//:${BSTINPUTS:-}"

cd "$DIR"
mkdir -p "$OUT"

echo "Using Springer template: $SPRINGER"
echo "Compiling main.tex with xelatex ..."

xelatex -interaction=nonstopmode main.tex || true
bibtex main || true
xelatex -interaction=nonstopmode main.tex || true
xelatex -interaction=nonstopmode main.tex || true

cp main.pdf "$OUT/SOICT2026.pdf"

echo ""
echo "PDF saved:"
echo "  $DIR/main.pdf"
echo "  $OUT/SOICT2026.pdf"
