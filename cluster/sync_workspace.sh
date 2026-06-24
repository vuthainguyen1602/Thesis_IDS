#!/bin/bash
# Rsync code + parquet data from Mac to both Jetsons (same IDS_ROOT path on workers)
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$CLUSTER_DIR/load_cluster_env.sh"

REMOTE_ROOT="${CLUSTER_DRIVER_IDS_ROOT:-/home/jetson/Thesis_IDS}"
LOCAL_ROOT="${IDS_MAC_ROOT:-$ROOT}"
LOCAL_DATA="${IDS_DATA_DIR:-$LOCAL_ROOT/data}"

RSYNC_EXCLUDES=(
    --exclude '.git'
    --exclude 'venv'
    --exclude 'raspberry/venv'
    --exclude '__pycache__'
    --exclude '.idea'
    --exclude 'data_smoke'
    --exclude '*.pyc'
    --exclude 'results'
)

sync_host() {
    local ssh_target="$1"
    echo ""
    echo "================================================================"
    echo "  Sync -> ${ssh_target}:${REMOTE_ROOT}"
    echo "================================================================"
    # shellcheck disable=SC2086
    ssh ${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10} "$ssh_target" "mkdir -p '${REMOTE_ROOT}'"
    # shellcheck disable=SC2086
    rsync -avz -e "ssh ${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}" --delete "${RSYNC_EXCLUDES[@]}" \
        "$LOCAL_ROOT/" "${ssh_target}:${REMOTE_ROOT}/"
}

if [ -z "${JETSON1_SSH:-}" ]; then
    echo "[ERR] Set JETSON1_SSH in cluster/spark_cluster.env"
    exit 1
fi

if [ "${JETSON2_ENABLED:-0}" = "1" ] && [ -z "${JETSON2_SSH:-}" ]; then
    echo "[ERR] JETSON2_ENABLED=1 but JETSON2_SSH is empty"
    exit 1
fi

if [ ! -d "${LOCAL_DATA}/train_data.parquet" ] || ! ls "${LOCAL_DATA}/train_data.parquet"/part-*.parquet 1>/dev/null 2>&1; then
    echo "[WARN] Parquet not found locally at ${LOCAL_DATA}/train_data.parquet"
    echo "       Run: python ml_00_prepare_cicids2017.py"
fi

echo "[INFO] Local:  ${LOCAL_ROOT}/"
echo "[INFO] Remote: ${REMOTE_ROOT}/"

sync_host "$JETSON1_SSH"
if [ "${JETSON2_ENABLED:-0}" = "1" ]; then
    sync_host "$JETSON2_SSH"
fi

echo ""
if [ "${JETSON2_ENABLED:-0}" = "1" ]; then
    echo "[OK] Workspace synced to both Jetsons at ${REMOTE_ROOT}"
else
    echo "[OK] Workspace synced to Jetson #1 at ${REMOTE_ROOT}"
fi
echo "     Cluster data path: ${IDS_CLUSTER_DATA_DIR:-${REMOTE_ROOT}/data}"
echo ""
