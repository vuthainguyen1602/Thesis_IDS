#!/usr/bin/env bash
# Find which package (in thesis.cls load order) breaks array's >{...} tabulars.
# Run from thesis/:  bash bisect_tabular.sh
set -u
cd "$(dirname "$0")"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Package loads in thesis.cls order (options preserved; slashbox needs path).
LOADS=(
  "\\usepackage{anyfontsize}"
  "\\usepackage{template/slashbox}"
  "\\usepackage{algpseudocode}"
  "\\usepackage{placeins}"
  "\\usepackage{fancyhdr}"
  "\\usepackage{geometry}"
  "\\usepackage{setspace}"
  "\\usepackage{titlesec}"
  "\\usepackage[titles]{tocloft}"
  "\\usepackage{graphicx}"
  "\\usepackage{caption,subcaption}"
  "\\usepackage{array,tabularx,booktabs,longtable}"
  "\\usepackage{multirow,makecell}"
  "\\usepackage{float}"
  "\\usepackage{ragged2e}"
  "\\usepackage{amsmath,amsfonts,amssymb,mathtools}"
  "\\usepackage{siunitx}"
  "\\usepackage[mathscr]{euscript}"
  "\\usepackage{pifont,adforn}"
  "\\usepackage{algorithm}"
  "\\usepackage{tikz}"
  "\\usepackage{pdfpages}"
  "\\usepackage{ftnxtra}"
  "\\usepackage{xcolor}"
  "\\usepackage{listings}"
  "\\usepackage{hyperref}"
  "\\usepackage{cleveref}"
  "\\usepackage{multido}"
  "\\usepackage{etoolbox}"
  "\\usepackage{times,latexsym,bbm,rotating}"
  "\\usepackage{enumitem,hhline,stmaryrd,bussproofs}"
  "\\usepackage{lipsum}"
  "\\usepackage{textcase}"
  "\\usepackage{acro,tabularray}"
)

run_probe() {  # $1 = number of loads to include
  local n=$1 doc="$WORK/probe.tex"
  {
    echo "\\documentclass[12pt]{report}"
    echo "\\usepackage[utf8]{inputenc}"
    echo "\\usepackage[T1,T5]{fontenc}"
    for ((i=0; i<n; i++)); do echo "${LOADS[$i]}"; done
    echo "\\begin{document}"
    echo "\\begin{tabular}{l@{\\hspace{10pt}}>{\\raggedright\\arraybackslash}p{10cm}}"
    echo "a & b \\\\"
    echo "\\end{tabular}"
    echo "\\end{document}"
  } > "$doc"
  ( pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$WORK" "$doc" >/dev/null 2>&1 )
}

TOTAL=${#LOADS[@]}
echo "[bisect] probe(0): fontenc/inputenc + array-less tabular baseline..."
# NOTE: probe(0) has no array package, so use a >-less tabular? No — the array
# line is at index 11; for n<=11 the tabular will fail for the WRONG reason.
# Instead always prepend the array line for n<12.
run_probe() {
  local n=$1 doc="$WORK/probe.tex"
  {
    echo "\\documentclass[12pt]{report}"
    echo "\\usepackage[utf8]{inputenc}"
    echo "\\usepackage[T1,T5]{fontenc}"
    echo "\\renewcommand{\\rmdefault}{ptm}"
    echo "\\usepackage{array}"
    for ((i=0; i<n; i++)); do echo "${LOADS[$i]}"; done
    echo "\\begin{document}"
    echo "\\begin{tabular}{l@{\\hspace{10pt}}>{\\raggedright\\arraybackslash}p{10cm}}"
    echo "a & b \\\\"
    echo "\\end{tabular}"
    echo "\\end{document}"
  } > "$doc"
  ( pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$WORK" "$doc" >/dev/null 2>&1 )
}

if ! run_probe 0; then
  echo "  probe(0) FAILS — even bare array is broken. Error:"
  grep -A4 "^!" "$WORK/probe.log" | head -12
  exit 1
fi
echo "  probe(0) OK"
if run_probe $TOTAL; then
  echo "[bisect] FULL preamble compiles fine — culprit is later cls code, not a package."
  exit 0
fi
lo=0; hi=$TOTAL
while (( hi - lo > 1 )); do
  mid=$(( (lo + hi) / 2 ))
  if run_probe $mid; then lo=$mid; echo "  probe($mid) -> OK"; else hi=$mid; echo "  probe($mid) -> FAIL"; fi
done
echo ""
echo "[bisect] CULPRIT = ${LOADS[$lo]}   (load index $lo — first load that breaks it)"
run_probe $hi || true
echo "--- actual error with the culprit included: ---"
grep -B2 -A6 "^!" "$WORK/probe.log" | head -20
