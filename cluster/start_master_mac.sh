#!/bin/bash
# Start Spark standalone master on Mac
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

if [ -f "$CLUSTER_DIR/spark_cluster.env" ]; then
    # shellcheck disable=SC1091
    source "$CLUSTER_DIR/spark_cluster.env"
fi

export JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home -v 17 2>/dev/null || true)}"
if [ -z "${JAVA_HOME:-}" ]; then
    echo "[ERR] JAVA_HOME not set. Install JDK 17: brew install openjdk@17"
    exit 1
fi

if [ -d "$ROOT/venv/bin" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/venv/bin/activate"
fi

export SPARK_HOME="${SPARK_HOME:-$(python -c "import pyspark; import os; print(os.path.dirname(pyspark.__path__[0]))")}"
MASTER_HOST="${SPARK_MASTER_HOST:-${MAC_IP:-$(ipconfig getifaddr en0 2>/dev/null || hostname)}}"
MASTER_PORT="${SPARK_MASTER_PORT:-7077}"
WEBUI_PORT="${SPARK_MASTER_WEBUI_PORT:-8080}"

echo "================================================================"
echo "  Spark Master (Mac)"
echo "  Host: ${MASTER_HOST}:${MASTER_PORT}"
echo "  UI:   http://${MASTER_HOST}:${WEBUI_PORT}"
echo "================================================================"

"$SPARK_HOME/sbin/start-master.sh" \
    --host "$MASTER_HOST" \
    --port "$MASTER_PORT" \
    --webui-port "$WEBUI_PORT"

echo ""
echo "Next on each Jetson:"
echo "  source cluster/spark_cluster.env && ./cluster/start_worker.sh"
echo ""
