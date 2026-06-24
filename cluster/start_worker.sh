#!/bin/bash
# Start Spark standalone worker (run on each Jetson Nano)
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

if [ -f "$CLUSTER_DIR/spark_cluster.env" ]; then
    # shellcheck disable=SC1091
    source "$CLUSTER_DIR/load_cluster_env.sh"
fi

# shellcheck disable=SC1091
source "$CLUSTER_DIR/resolve_spark_home.sh"

export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}"

if [ -d "$ROOT/jetson/venv/bin" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/jetson/venv/bin/activate"
elif [ -d "$ROOT/venv/bin" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/venv/bin/activate"
fi

export_spark_home

MASTER_URL="${SPARK_MASTER:?Set SPARK_MASTER in cluster/spark_cluster.env}"
NODE_ID="${EDGE_NODE_ID:-$(hostname)}"
CORES="${SPARK_WORKER_CORES:-4}"
MEMORY="${SPARK_WORKER_MEMORY:-5g}"

echo "[INFO] Spark worker ($NODE_ID) -> $MASTER_URL (${CORES} cores, ${MEMORY})"
echo "[INFO] SPARK_HOME=$SPARK_HOME"

start_spark_worker "$MASTER_URL" "$CORES" "$MEMORY"
echo "[OK] Worker started"
