#!/bin/bash
# Compile thesis (report + XeLaTeX + Springer splncs04 bibliography)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
SPRINGER="$ROOT/Latex-Template-for-Springer"
BIB="$ROOT/papers/latex"
OUT="$ROOT/output/pdfs"

export BSTINPUTS="${BIB}//:${SPRINGER}//:${BSTINPUTS:-}"

cd "$DIR"
mkdir -p "$OUT"

echo "Using Springer bib style: $SPRINGER/splncs04.bst"
echo "Compiling main.tex with xelatex ..."

xelatex -interaction=nonstopmode main.tex || true
bibtex main || true
xelatex -interaction=nonstopmode main.tex || true
xelatex -interaction=nonstopmode main.tex || true

cp main.pdf "$OUT/thesis_draft.pdf"

echo ""
echo "PDF saved:"
echo "  $DIR/main.pdf"
echo "  $OUT/thesis_draft.pdf"
