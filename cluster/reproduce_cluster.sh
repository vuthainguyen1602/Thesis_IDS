#!/bin/bash
# Full distributed reproduction: Mac master + Jetson workers + remote driver
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$CLUSTER_DIR/load_cluster_env.sh"

TRACK="${1:-fair}"

run_local_mac() {
    cd "$ROOT"
    if [ -d venv/bin ]; then
        # shellcheck disable=SC1091
        source venv/bin/activate
    fi
    export IDS_ROOT="$ROOT"
    export IDS_DATA_DIR="${IDS_DATA_DIR:-$ROOT/data}"
    export IDS_ALLOW_LOCAL_SPARK=1
    unset IDS_SPARK_CLUSTER SPARK_MASTER SPARK_DRIVER_HOST
    python "$@"
}

run_remote() {
    "$CLUSTER_DIR/run_ml_remote.sh" "$@"
}

echo "================================================================"
echo "  Distributed cluster reproduce (track=$TRACK)"
echo "  Master: ${SPARK_MASTER}"
echo "  Driver: ${CLUSTER_DRIVER}"
echo "  Data (workers): ${IDS_CLUSTER_DATA_DIR}"
echo "================================================================"

"$CLUSTER_DIR/check_cluster.sh" || {
    echo "[INFO] Starting Spark master on Mac..."
    "$CLUSTER_DIR/start_master_mac.sh"
    sleep 3
}

if [ ! -f "${IDS_DATA_DIR:-$ROOT/data}/train_data.parquet" ]; then
    echo "[1] ml_00 on Mac (local Spark — raw CSV on Mac only)..."
    run_local_mac ml_00_prepare_cicids2017.py
else
    echo "[1] Parquet OK on Mac — skip ml_00"
fi

echo "[2] Sync workspace to Jetsons..."
"$CLUSTER_DIR/sync_workspace.sh"

case "$TRACK" in
  fair)
    run_remote ml_01_baseline_all_features.py
    run_remote ml_02_feature_selection_rf.py
    run_remote ml_04_dimensionality_reduction_pca.py
    run_remote ml_05_shap_explainability.py
    run_remote ml_06_feature_selection_shap.py
    run_remote ml_07_cross_method_comparison.py
    echo ""
    echo "Done FAIR track. Run: ./papers/fair2026/collect_results.sh"
    ;;
  soict)
    run_remote ml_08_anomaly_gate_autoencoder.py
    run_local_mac raspberry/scripts/save_model.py
    "$CLUSTER_DIR/sync_workspace.sh"
    echo ""
    echo "Done SOICT prep. Model synced. Start Docker + Jetson edge pipelines."
    ;;
  thesis)
    run_remote ml_01_baseline_all_features.py
    run_remote ml_02_feature_selection_rf.py
    run_remote ml_04_dimensionality_reduction_pca.py
    run_remote ml_05_shap_explainability.py
    run_remote ml_06_feature_selection_shap.py
    run_remote ml_07_cross_method_comparison.py
    run_remote ml_03_hyperparameter_tuning.py
    run_remote ml_08_anomaly_gate_autoencoder.py
    run_local_mac raspberry/scripts/save_model.py
    "$CLUSTER_DIR/sync_workspace.sh"
    echo ""
    echo "Done thesis track. Model synced to Jetsons."
    ;;
  *)
    echo "Usage: $0 {fair|soict|thesis}"
    exit 1
    ;;
esac
