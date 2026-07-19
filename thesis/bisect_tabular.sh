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
echo "[bisect] baseline (array only up to index 11)..."
run_probe 12 && echo "  baseline OK" || { echo "  BASELINE FAILS — array itself broken?!"; exit 1; }

# quick full check
if run_probe $TOTAL; then
  echo "[bisect] FULL preamble compiles fine — culprit is NOT a package load; it must be later cls code."
  exit 0
fi
# invariant: probe(lo)=OK, probe(hi)=FAIL
lo=12; hi=$TOTAL
while (( hi - lo > 1 )); do
  mid=$(( (lo + hi) / 2 ))
  if run_probe $mid; then lo=$mid; else hi=$mid; fi
  echo "  probe($mid) -> $( (( mid == lo )) && echo OK || echo FAIL )"
done
echo ""
echo "[bisect] CULPRIT = ${LOADS[$lo]}   (load index $lo — first one that breaks the tabular)"
