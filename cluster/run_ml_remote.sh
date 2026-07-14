#!/bin/bash
# Run one ML script on Jetson #1 driver against Mac Spark master
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

source "$CLUSTER_DIR/load_cluster_env.sh"

SCRIPT="${1:-}"
if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 <ml_script.py> [extra args...]"
    exit 1
fi
shift || true

DRIVER="${CLUSTER_DRIVER:?Set CLUSTER_DRIVER in cluster/spark_cluster.env}"
REMOTE_ROOT="${CLUSTER_DRIVER_IDS_ROOT:-/home/jetson/Thesis_IDS}"
SCRIPT_BASE="$(basename "$SCRIPT")"

if [ ! -f "$ROOT/$SCRIPT" ] && [ ! -f "$ROOT/$SCRIPT_BASE" ]; then
    echo "[ERR] Script not found: $SCRIPT"
    exit 1
fi

echo "================================================================"
echo "  Distributed ML run"
echo "  Driver:  $DRIVER"
echo "  Master:  ${SPARK_MASTER:?}"
echo "  Script:  $SCRIPT_BASE"
echo "================================================================"

SSH_OPTS="${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
INSTALL_SCRIPT="$CLUSTER_DIR/install_ml_deps.sh"

install_ml_deps_remote() {
    local host="$1"
    echo ""
    echo "[INFO] Ensuring ML deps on executor host: $host"
    ssh $SSH_OPTS "$host" "IDS_ROOT='$REMOTE_ROOT' bash -s" < "$INSTALL_SCRIPT"
}

# Executors run Python on every Spark worker — not only the driver.
install_ml_deps_remote "$DRIVER"
if [ "${JETSON2_ENABLED:-0}" = "1" ] && [ -n "${JETSON2_SSH:-}" ]; then
    install_ml_deps_remote "$JETSON2_SSH"
fi

ssh $SSH_OPTS "$DRIVER" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -d jetson/venv/bin ]; then source jetson/venv/bin/activate; elif [ -d venv/bin ]; then source venv/bin/activate; fi
python -c "import pandas, matplotlib, seaborn, pyarrow, xgboost, shap; print('[OK] Core ML deps ready on driver')"
export JAVA_HOME="\${JAVA_HOME:-\$(dirname "\$(dirname "\$(readlink -f "\$(which java)")")")}"
export PATH="\$JAVA_HOME/bin:\$PATH"
echo "[INFO] JAVA_HOME=\$JAVA_HOME"
export IDS_SPARK_CLUSTER=1
export IDS_ROOT="$REMOTE_ROOT"
export IDS_CLUSTER_DATA_DIR="${IDS_CLUSTER_DATA_DIR:-$REMOTE_ROOT/data}"
export SPARK_MASTER="${SPARK_MASTER}"
export SPARK_DRIVER_HOST="${SPARK_DRIVER_HOST:-$(echo "$DRIVER" | cut -d@ -f2)}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-3g}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-3g}"
export SPARK_DRIVER_MAX_RESULT_SIZE="${SPARK_DRIVER_MAX_RESULT_SIZE:-3g}"
export SPARK_EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-4}"
export SPARK_SHUFFLE_PARTITIONS="${SPARK_SHUFFLE_PARTITIONS:-32}"
unset IDS_ALLOW_LOCAL_SPARK
export IDS_EXP7_START_STEP="${IDS_EXP7_START_STEP:-1}"
export IDS_EXP7_AGGREGATE_ONLY="${IDS_EXP7_AGGREGATE_ONLY:-0}"
python "$SCRIPT_BASE" "$@"
EOF

echo ""
echo "[OK] Finished: $SCRIPT_BASE"
echo ""
