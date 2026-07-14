#!/bin/bash
# Stop Spark master (Mac) and workers (Jetson via SSH)
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$CLUSTER_DIR/spark_cluster.env" ]; then
    source "$CLUSTER_DIR/load_cluster_env.sh"
fi

export SPARK_HOME="${SPARK_HOME:-$(python -c "import pyspark; print(pyspark.__path__[0])" 2>/dev/null || echo "")}"

# BatchMode=yes = no password prompt; use ssh-copy-id jetson@<IP> once on Mac.
SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes)

stop_local() {
    if [ -n "$SPARK_HOME" ] && [ -f "$SPARK_HOME/sbin/stop-master.sh" ]; then
        unset SPARK_MASTER
        "$SPARK_HOME/sbin/stop-master.sh" 2>/dev/null || true
        "$SPARK_HOME/sbin/stop-worker.sh" 2>/dev/null || true
    fi
}

stop_remote_worker() {
    local host="$1"
    echo "[INFO] Stopping worker on ${host} ..."
    if "${SSH_CMD[@]}" "$host" \
        "export SPARK_HOME=\$(python3 -c \"import pyspark; print(pyspark.__path__[0])\" 2>/dev/null); \
        unset SPARK_MASTER; \
        [ -f \"\$SPARK_HOME/sbin/stop-worker.sh\" ] && \$SPARK_HOME/sbin/stop-worker.sh"; then
        echo "[OK] Worker stop sent to ${host}"
    else
        echo "[WARN] Could not stop worker on ${host}"
        echo "       Run once on Mac: ssh-copy-id ${host}"
    fi
}

stop_local

if [ -n "${JETSON1_SSH:-}" ]; then
    stop_remote_worker "${JETSON1_SSH}"
fi

if [ "${JETSON2_ENABLED:-0}" = "1" ] && [ -n "${JETSON2_SSH:-}" ]; then
    stop_remote_worker "${JETSON2_SSH}"
else
    echo "[INFO] Jetson #2 skipped (JETSON2_ENABLED=${JETSON2_ENABLED:-0})"
fi

echo "[OK] Cluster stop requested"
