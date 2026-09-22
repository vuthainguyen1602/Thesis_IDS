#!/bin/bash
# Build all manuscripts and copy PDFs to output/pdfs/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/output/pdfs"
mkdir -p "$OUT"

echo "Building FAIR'2026 (English)..."
"$ROOT/papers/fair2026/manuscript/compile.sh" || true
cp "$ROOT/papers/fair2026/manuscript/main_en.pdf" "$OUT/FAIR2026.pdf"

echo "Building SOICT 2026..."
"$ROOT/papers/soict2026/manuscript/compile.sh" || true
cp "$ROOT/papers/soict2026/manuscript/main.pdf" "$OUT/SOICT2026.pdf"

echo "Building thesis..."
(cd "$ROOT/thesis" && latexmk -pdf -interaction=nonstopmode main.tex) || true
cp "$ROOT/thesis/main.pdf" "$OUT/LuanVan_ThacSi.pdf"

echo ""
echo "PDF files:"
ls -la "$OUT"/*.pdf
echo ""
echo "Open folder: $OUT"
