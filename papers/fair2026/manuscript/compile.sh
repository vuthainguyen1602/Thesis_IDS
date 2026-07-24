#!/bin/bash
# Compile FAIR'2026 paper (Springer LNCS + XeLaTeX for Vietnamese)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
IEEETPL="$ROOT/papers/fair2026/IEEE-conference-template-062824"
BIB="$ROOT/papers/latex"
OUT="$ROOT/output/pdfs"

export TEXINPUTS="${IEEETPL}//:${TEXINPUTS:-}"
export BSTINPUTS="${BIB}//:${IEEETPL}//:${BSTINPUTS:-}"

cd "$DIR"
mkdir -p "$OUT"

echo "Using IEEEtran template: $IEEETPL"
echo "Compiling main_en.tex with xelatex ..."

xelatex -interaction=nonstopmode main_en.tex || true
bibtex main_en || true
xelatex -interaction=nonstopmode main_en.tex || true
xelatex -interaction=nonstopmode main_en.tex || true

cp main_en.pdf "$OUT/FAIR2026.pdf"

echo ""
echo "PDF saved:"
echo "  $DIR/main_en.pdf"
echo "  $OUT/FAIR2026.pdf"
