#!/bin/bash
# Rsync code + parquet data from Mac to both Jetsons (same IDS_ROOT path on workers)
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$CLUSTER_DIR/load_cluster_env.sh"

REMOTE_ROOT="${CLUSTER_DRIVER_IDS_ROOT:-/home/jetson/Thesis_IDS}"
LOCAL_ROOT="${IDS_ROOT:-$ROOT}"

RSYNC_EXCLUDES=(
    --exclude '.git'
    --exclude 'venv'
    --exclude 'raspberry/venv'
    --exclude '__pycache__'
    --exclude '.idea'
    --exclude 'data_smoke'
    --exclude '*.pyc'
)

sync_host() {
    local ssh_target="$1"
    echo ""
    echo "================================================================"
    echo "  Sync -> ${ssh_target}:${REMOTE_ROOT}"
    echo "================================================================"
    ssh "$ssh_target" "mkdir -p '${REMOTE_ROOT}'"
    rsync -avz --delete "${RSYNC_EXCLUDES[@]}" \
        "$LOCAL_ROOT/" "${ssh_target}:${REMOTE_ROOT}/"
}

if [ -z "${JETSON1_SSH:-}" ] || [ -z "${JETSON2_SSH:-}" ]; then
    echo "[ERR] Set JETSON1_SSH and JETSON2_SSH in cluster/spark_cluster.env"
    exit 1
fi

if [ ! -f "${IDS_DATA_DIR:-$LOCAL_ROOT/data}/train_data.parquet" ]; then
    echo "[WARN] Parquet not found locally. Run ml_00_prepare_cicids2017.py on Mac first."
fi

sync_host "$JETSON1_SSH"
sync_host "$JETSON2_SSH"

echo ""
echo "[OK] Workspace synced to both Jetsons at ${REMOTE_ROOT}"
echo "     Cluster data path: ${IDS_CLUSTER_DATA_DIR:-${REMOTE_ROOT}/data}"
echo ""
