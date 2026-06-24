#!/bin/bash
# Start Spark standalone worker on Jetson #2 (or Jetson #1 as co-worker)
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

export JAVA_HOME="${JAVA_HOME:-$(dirname $(dirname $(readlink -f $(which java))))}"
export SPARK_HOME="${SPARK_HOME:-$(python -c "import pyspark; print(pyspark.__path__[0])")}"
WORKER_SCRIPT="${SPARK_HOME}/sbin/start-worker.sh"
if [ ! -f "${WORKER_SCRIPT}" ]; then
    WORKER_SCRIPT="${SPARK_HOME}/sbin/start-slave.sh"
fi
MASTER_URL="${SPARK_MASTER:-spark://192.168.1.165:7077}"
WORKER_CORES="${SPARK_WORKER_CORES:-4}"
WORKER_MEMORY="${SPARK_WORKER_MEMORY:-5g}"

echo "[INFO] Starting Spark worker -> ${MASTER_URL}"
# Avoid Spark sbin rsync hook (SPARK_MASTER env = host:path, not spark:// URL).
unset SPARK_MASTER
"${WORKER_SCRIPT}" \
    --cores "${WORKER_CORES}" \
    --memory "${WORKER_MEMORY}" \
    "${MASTER_URL}"

echo "[OK] Spark worker connected to ${MASTER_URL}"
