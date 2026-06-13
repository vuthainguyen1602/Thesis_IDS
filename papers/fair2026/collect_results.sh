#!/bin/bash
# Copy FAIR paper tables and reports
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
RES="$PAPER_DIR/results"
TABLES="$PAPER_DIR/tables"
ML07="$ROOT/results/ml_07_cross_method_comparison"
ML02="$ROOT/results/ml_02_feature_selection_rf"
SHARED="$ROOT/results/shared"

mkdir -p "$RES" "$TABLES"

copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -e "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo "  copied: $(basename "$src")"
    fi
}

echo "Collecting FAIR'2026 tables and reports..."

copy_if_exists "$ML02/feature_importance.csv" "$TABLES/feature_importance.csv"
copy_if_exists "$ML07/cross_method_summary.csv" "$TABLES/cross_method_summary.csv"
copy_if_exists "$ML07/drift_simulation_summary.csv" "$TABLES/drift_simulation_summary.csv"
copy_if_exists "$ML07/robustness_holdout_summary.csv" "$TABLES/robustness_holdout_summary.csv"
copy_if_exists "$ML07/statistical_validity_multiseed.csv" "$TABLES/statistical_validity_multiseed.csv"
copy_if_exists "$SHARED/best_config.json" "$RES/best_config.json"
copy_if_exists "$ML07/report.html" "$RES/ml_07_report.html"

echo ""
echo "Figures: use ../../../results/ml_07_cross_method_comparison/ from manuscript."
echo "  tables/  -> $TABLES"
echo "  results/ -> $RES"
