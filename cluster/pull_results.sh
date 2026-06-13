#!/bin/bash
# Pull ML results (and optional edge models) from Jetson #1 driver to Mac
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$CLUSTER_DIR/load_cluster_env.sh"

REMOTE="${CLUSTER_DRIVER:?Set CLUSTER_DRIVER in cluster/spark_cluster.env}"
REMOTE_ROOT="${CLUSTER_DRIVER_IDS_ROOT:-/home/jetson/Thesis_IDS}"
LOCAL_ROOT="${IDS_MAC_ROOT:-$ROOT}"

PULL_MODELS="${PULL_MODELS:-1}"

RSYNC_OPTS=(-avz -e "ssh ${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}")

echo "================================================================"
echo "  Pull results: ${REMOTE}:${REMOTE_ROOT} -> ${LOCAL_ROOT}"
echo "================================================================"

mkdir -p "${LOCAL_ROOT}/results"

# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" \
    "${REMOTE}:${REMOTE_ROOT}/results/" \
    "${LOCAL_ROOT}/results/"

echo "[OK] results/ synced to ${LOCAL_ROOT}/results/"

if [ "$PULL_MODELS" = "1" ]; then
    mkdir -p "${LOCAL_ROOT}/raspberry/model"
    # shellcheck disable=SC2086
    rsync "${RSYNC_OPTS[@]}" \
        "${REMOTE}:${REMOTE_ROOT}/raspberry/model/" \
        "${LOCAL_ROOT}/raspberry/model/"
    echo "[OK] raspberry/model/ synced"
fi

echo ""
echo "Next (optional):"
echo "  ./papers/fair2026/collect_results.sh"
echo "  ./thesis/collect_results.sh"
echo ""
