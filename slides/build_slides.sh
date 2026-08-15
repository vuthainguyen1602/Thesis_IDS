#!/bin/bash
# Build the presentation decks in slides/ and copy the PDFs to output/pdfs/.
#
#   ./build_slides.sh              # build all three decks
#   ./build_slides.sh soict2026    # build one deck
#   NOTES=1 ./build_slides.sh      # also build the speaker-notes versions
#
# XeLaTeX is required (fontspec + Vietnamese diacritics); beamer's metropolis
# theme and appendixnumberbeamer must be installed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLIDES="$ROOT/slides"
OUT="$ROOT/output/pdfs"
mkdir -p "$OUT"

# deck directory -> output PDF basename
deck_output() {
  case "$1" in
    soict2026) echo "SOICT2026_slides" ;;
    fair2026)  echo "FAIR2026_slides" ;;
    defense)   echo "LuanVan_BaoVe_slides" ;;
    *)         echo "$1" ;;
  esac
}

build_deck() {
  local deck="$1"
  local dir="$SLIDES/$deck"
  local name; name="$(deck_output "$deck")"

  [ -f "$dir/main.tex" ] || { echo "!! no main.tex in $dir"; return 1; }

  echo "==> Building $deck"
  cd "$dir"
  # Two passes: beamer needs the second one for the frame-count fraction in the
  # footer and for \appendix-aware numbering.
  xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  cp main.pdf "$OUT/$name.pdf"
  echo "    $OUT/$name.pdf"

  if [ "${NOTES:-0}" = "1" ]; then
    echo "==> Building $deck (speaker notes)"
    # \def\shownotes{} switches the preamble to "notes on second screen".
    xelatex -interaction=nonstopmode -halt-on-error -jobname=main_notes \
      "\def\shownotes{}\input{main.tex}" >/dev/null
    xelatex -interaction=nonstopmode -halt-on-error -jobname=main_notes \
      "\def\shownotes{}\input{main.tex}" >/dev/null
    cp main_notes.pdf "$OUT/${name}_notes.pdf"
    echo "    $OUT/${name}_notes.pdf"
  fi
}

if [ $# -gt 0 ]; then
  for d in "$@"; do build_deck "$d"; done
else
  for d in soict2026 fair2026 defense; do build_deck "$d"; done
fi

echo ""
echo "Slide PDFs in $OUT:"
ls -la "$OUT"/*slides*.pdf
