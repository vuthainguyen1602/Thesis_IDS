#!/bin/bash
# Build the SOICT 2026 A0 poster and export it for the organisers (PDF + PPTX).
# Not part of build_slides.sh: the poster is a single A0 page, not a Beamer deck.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$here"
# Two passes: the header/footer are TikZ overlays on "current page", whose
# position comes from the .aux; one pass in a clean checkout drops the footer.
for pass in 1 2; do xelatex -interaction=nonstopmode main.tex >/dev/null || true; done
if grep -q '^! ' main.log; then grep -A3 '^! ' main.log; exit 1; fi
pages=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
[ "$pages" = 1 ] || { echo "poster spilled onto $pages pages" >&2; exit 1; }
grep 'POSTER' main.log
cp main.pdf "$root/output/pdfs/SOICT2026_poster.pdf"
python3 "$root/slides/pdf2pptx.py" main.pdf "$root/output/pdfs/SOICT2026_poster.pptx" --dpi 200 --no-notes
