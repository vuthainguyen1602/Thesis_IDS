#!/bin/bash
# Copy the auto-generated experiment figures into thesis/img/ so the thesis
# picks up the LATEST (leakage-aware) plots. Run after re-running the pipeline
# and pulling results back to the Mac (./cluster/pull_results.sh).
#
# Usage:  ./thesis/collect_figures.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"      # thesis/
ROOT="$(cd "$DIR/.." && pwd)"             # repo root
IMG="$DIR/img"
RES="$ROOT/results"
mkdir -p "$IMG"

copy() {  # copy <src> <dst-name>
  if [ -f "$1" ]; then
    cp -f "$1" "$IMG/$2"
    echo "  [OK]   $2  <-  ${1#$ROOT/}"
  else
    echo "  [MISS] $2  (source not found: ${1#$ROOT/} — run the script first)"
  fi
}

echo "Collecting thesis figures into thesis/img/ ..."

# Script-generated figures (names already match what the thesis imports)
copy "$RES/ml_06_feature_selection_shap/shap_feature_importance_top30.png" "shap_feature_importance_top30.png"
copy "$RES/ml_06_feature_selection_shap/shap_summary_beeswarm.png"          "shap_summary_beeswarm.png"
copy "$RES/ml_09_multiclass_eval/confusion_matrix.png"                     "confusion_matrix.png"

echo
echo "Manual figures (NOT auto-generated — update by hand if needed):"
echo "  - benchmark_comparison.png / model_tradeoff_radar.png"
echo "      build them from jetson/model/benchmark_comparison.json (jetson/scripts/benchmark_all.py)"
echo "  - pipeline_terminal / grafana_dashboard / grafana_attacks / mailtrap_alert / slack_alert"
echo "      these are live-system screenshots — capture and drop into thesis/img/"
echo
echo "Done. Figures missing from thesis/img/ render as a red placeholder box (\\IfFileExists)."
echo
echo "WARNING: img/legacy/ contains the OLD Raspberry-Pi-era figures (wrong hardware,"
echo "pre-leakage-fix results — the old SHAP plot even ranks destination_port #2)."
echo "NEVER copy them back. Re-capture everything on the Jetson Orin Nano Super cluster"
echo "with the leak-free SHAP Top-30 model; hostnames/series in screenshots must show"
echo "the Jetson nodes, not 'raspberrypi'."
