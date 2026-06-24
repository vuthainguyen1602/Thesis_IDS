#!/bin/bash
# Start Spark standalone master on Jetson #1
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

export JAVA_HOME="${JAVA_HOME:-$(dirname $(dirname $(readlink -f $(which java))))}"
export SPARK_HOME="${SPARK_HOME:-$(python -c "import pyspark; print(pyspark.__path__[0])")}"
MASTER_SCRIPT="${SPARK_HOME}/sbin/start-master.sh"
MASTER_HOST="${SPARK_MASTER_HOST:-$(hostname -I | awk '{print $1}')}"
MASTER_PORT="${SPARK_MASTER_PORT:-7077}"
WEBUI_PORT="${SPARK_MASTER_WEBUI_PORT:-8080}"

echo "[INFO] Starting Spark master on ${MASTER_HOST}:${MASTER_PORT}"
"${MASTER_SCRIPT}" \
    --host "${MASTER_HOST}" \
    --port "${MASTER_PORT}" \
    --webui-port "${WEBUI_PORT}"

echo "[OK] Spark master UI: http://${MASTER_HOST}:${WEBUI_PORT}"
echo "Set on both Jetsons: SPARK_MASTER=spark://${MASTER_HOST}:${MASTER_PORT}"
