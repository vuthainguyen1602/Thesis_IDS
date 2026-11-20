#!/bin/bash
# Natural column heights: same poster, columns without the fixed height and
# with fixed 8 mm gaps instead of \vfill. Prints the POSTER lines.
set -euo pipefail
cd "$(dirname "$0")"
sed -e 's/\[t\]\[\\colheight\]\[t\]{381mm}/[t]{381mm}/' -e 's/^\\vfill$/\\vspace{8mm}/' main.tex > _measure.tex
xelatex -interaction=nonstopmode _measure.tex >/dev/null 2>&1 || true
grep -A2 '^! ' _measure.log || true
grep POSTER _measure.log
rm -f _measure.*
