#!/bin/bash
# Collect SOICT paper artifacts
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
FIG="$PAPER_DIR/figures"
RES="$PAPER_DIR/results"
TABLES="$PAPER_DIR/tables"

mkdir -p "$FIG" "$RES/benchmarks" "$TABLES"

copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -e "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -R "$src" "$dst"
        echo "  copied: $(basename "$src")"
    fi
}

echo "Collecting SOICT 2026 artifacts..."

# Architecture doc → figures reference
copy_if_exists "$ROOT/raspberry/JETSON_DISTRIBUTED.md" "$RES/JETSON_DISTRIBUTED.md"

# Model metadata
copy_if_exists "$ROOT/raspberry/model/feature_columns.json" "$TABLES/feature_columns.json"
copy_if_exists "$ROOT/raspberry/model/models_info.json" "$TABLES/models_info.json"
copy_if_exists "$ROOT/raspberry/model/anomaly_threshold.json" "$TABLES/anomaly_threshold.json"

# Env templates (for reproducibility appendix)
copy_if_exists "$ROOT/raspberry/.env.jetson1.example" "$RES/env.jetson1.example"
copy_if_exists "$ROOT/raspberry/.env.jetson2.example" "$RES/env.jetson2.example"
copy_if_exists "$ROOT/raspberry/.env.jetson-horizontal.example" "$RES/env.jetson-horizontal.example"

# Benchmark outputs (if run_benchmarks.sh was executed)
copy_if_exists "$PAPER_DIR/results/benchmarks/summary.csv" "$TABLES/benchmark_summary.csv"
copy_if_exists "$PAPER_DIR/results/benchmarks/summary_template.csv" "$TABLES/benchmark_summary_template.csv"
if [ -d "$PAPER_DIR/results/benchmarks" ]; then
    echo "  benchmarks/ already in place"
fi

# Grafana dashboard export (optional)
copy_if_exists "$ROOT/raspberry/dashboard/grafana_dashboard.json" "$FIG/grafana_dashboard.json"

echo ""
echo "Done."
echo "  Add architecture diagram to figures/architecture.png (draw manually)"
echo "  Add benchmark JSON/CSV to results/benchmarks/ after Jetson runs"
