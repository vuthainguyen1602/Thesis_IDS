#!/bin/bash
# Collect thesis artifacts
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RES="$ROOT/thesis/results"
ML07="$ROOT/results/ml_07_cross_method_comparison"
ML02="$ROOT/results/ml_02_feature_selection_rf"
SHARED="$ROOT/results/shared"

mkdir -p "$RES"

copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -e "$src" ]; then
        cp "$src" "$dst"
        echo "  copied: $(basename "$src")"
    fi
}

echo "Collecting thesis results..."

copy_if_exists "$ML02/feature_importance.csv" "$RES/feature_importance.csv"
copy_if_exists "$SHARED/best_config.json" "$RES/best_config.json"
copy_if_exists "$ML07/cross_method_summary.csv" "$RES/cross_method_summary.csv"
copy_if_exists "$ML07/drift_simulation_summary.csv" "$RES/drift_simulation_summary.csv"
copy_if_exists "$ML07/robustness_holdout_summary.csv" "$RES/robustness_holdout_summary.csv"
copy_if_exists "$ML07/statistical_validity_multiseed.csv" "$RES/statistical_validity_multiseed.csv"
copy_if_exists "$ML07/report.html" "$RES/ml_07_report.html"

echo "Done -> $RES"
