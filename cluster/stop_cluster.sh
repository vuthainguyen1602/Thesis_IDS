#!/bin/bash
# Stop Spark master (Mac) and workers (Jetson via SSH)
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

if [ -f "$CLUSTER_DIR/spark_cluster.env" ]; then
    # shellcheck disable=SC1091
    source "$CLUSTER_DIR/spark_cluster.env"
fi

export SPARK_HOME="${SPARK_HOME:-$(python -c "import pyspark; import os; print(os.path.dirname(pyspark.__path__[0]))" 2>/dev/null || echo "")}"

stop_local() {
    if [ -n "$SPARK_HOME" ] && [ -f "$SPARK_HOME/sbin/stop-master.sh" ]; then
        "$SPARK_HOME/sbin/stop-master.sh" 2>/dev/null || true
        "$SPARK_HOME/sbin/stop-worker.sh" 2>/dev/null || true
    fi
}

stop_local

for host in "${JETSON1_SSH:-}" "${JETSON2_SSH:-}"; do
    [ -z "$host" ] && continue
    echo "[INFO] Stopping worker on $host ..."
    ssh "$host" "export SPARK_HOME=\$(python3 -c \"import pyspark,os; print(os.path.dirname(pyspark.__path__[0]))\" 2>/dev/null); \
        [ -f \"\$SPARK_HOME/sbin/stop-worker.sh\" ] && \$SPARK_HOME/sbin/stop-worker.sh" 2>/dev/null || true
done

echo "[OK] Cluster stop requested"
