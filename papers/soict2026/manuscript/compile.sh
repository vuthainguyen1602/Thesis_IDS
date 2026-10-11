#!/bin/bash
# Compile SOICT 2026 paper (Springer LNCS + XeLaTeX). The folder is
# self-contained; figures are refreshed from their sources when present.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
OUT="$ROOT/output/pdfs"

cd "$DIR"
mkdir -p "$OUT" figures

for src in \
  "$ROOT/papers/soict2026/figures/grafana_dashboard.png" \
  "$ROOT/papers/soict2026/figures/grafana_peak_load.png" \
  "$ROOT/papers/soict2026/results/remeasure_20260930/edge_capacity.png" \
  "$ROOT/results/ml_08_anomaly_gate/gate_operating_points.png"; do
  [ -f "$src" ] && cp "$src" figures/
done

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
