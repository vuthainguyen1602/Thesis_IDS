#!/bin/bash
# Start Spark standalone worker (run on each Jetson Nano)
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

if [ -f "$CLUSTER_DIR/spark_cluster.env" ]; then
    # shellcheck disable=SC1091
    source "$CLUSTER_DIR/load_cluster_env.sh"
fi

export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}"

if [ -d "$ROOT/raspberry/venv/bin" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/raspberry/venv/bin/activate"
elif [ -d "$ROOT/venv/bin" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/venv/bin/activate"
fi

export SPARK_HOME="${SPARK_HOME:-$(python -c "import pyspark; import os; print(os.path.dirname(pyspark.__path__[0]))")}"
MASTER_URL="${SPARK_MASTER:?Set SPARK_MASTER in cluster/spark_cluster.env}"
WORKER_SCRIPT="$SPARK_HOME/sbin/start-worker.sh"
if [ ! -f "$WORKER_SCRIPT" ]; then
    WORKER_SCRIPT="$SPARK_HOME/sbin/start-slave.sh"
fi

NODE_ID="${EDGE_NODE_ID:-$(hostname)}"
CORES="${SPARK_WORKER_CORES:-2}"
MEMORY="${SPARK_WORKER_MEMORY:-768m}"

echo "[INFO] Spark worker ($NODE_ID) -> $MASTER_URL (${CORES} cores, ${MEMORY})"
"$WORKER_SCRIPT" --cores "$CORES" --memory "$MEMORY" "$MASTER_URL"
echo "[OK] Worker started"
