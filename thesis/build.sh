#!/bin/bash
# Build the thesis, refusing to embed a stale copy of the paper.
#
# thesis/main.tex attaches ../output/pdfs/FAIR2026.pdf as the appendix, so the
# thesis carries whatever version of the paper sits there. Building the thesis
# alone after editing the paper silently ships the old one: no error, no warning,
# just an appendix that disagrees with the chapter citing it. This wrapper checks
# for that before running LaTeX.
#
#   ./thesis/build.sh          # check, build, copy to output/pdfs/
#   ./thesis/build.sh --force  # build even if the attached paper looks stale
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/papers/fair2026/manuscript/main_en.pdf"
EMB="$ROOT/output/pdfs/FAIR2026.pdf"

if [ -f "$SRC" ] && [ -f "$EMB" ] && [ "$SRC" -nt "$EMB" ]; then
    echo "[WARN] $SRC is newer than the attached $EMB."
    echo "       The appendix would carry the older paper. Refresh it with:"
    echo "         cp papers/fair2026/manuscript/main_en.pdf output/pdfs/FAIR2026.pdf"
    [ "${1:-}" = "--force" ] || { echo "[ERR] refusing to build (pass --force to override)"; exit 1; }
fi
if [ ! -f "$EMB" ]; then
    echo "[WARN] $EMB is missing — the appendix will fall back to a placeholder box."
fi

cd "$ROOT/thesis"
latexmk -pdf -interaction=nonstopmode main.tex
pages=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
# -a is not optional: this log carries Vietnamese chapter titles, and the grep on
# this machine (ugrep) classifies it as binary and reports no matches at all,
# which hid five overfull boxes — one of them 97pt — behind a clean "0".
over=$(grep -ao "Overfull" main.log | wc -l | tr -d " ")
worst=$(grep -ao "Overfull \\\\hbox ([0-9.]*pt" main.log | sed 's/.*(//' | sort -rn | head -1)
echo "[OK] thesis: ${pages} pages, ${over} overfull box(es)${worst:+, worst ${worst}pt}"
mkdir -p "$ROOT/output/pdfs"
cp main.pdf "$ROOT/output/pdfs/LuanVan_ThacSi.pdf"
echo "[OK] copied to output/pdfs/LuanVan_ThacSi.pdf"
