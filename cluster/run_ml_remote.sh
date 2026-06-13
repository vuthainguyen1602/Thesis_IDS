#!/bin/bash
# Run one ML script on Jetson #1 driver against Mac Spark master
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

# shellcheck disable=SC1091
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

# shellcheck disable=SC2086
ssh ${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10} "$DRIVER" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -d raspberry/venv/bin ]; then source raspberry/venv/bin/activate; elif [ -d venv/bin ]; then source venv/bin/activate; fi
if ! python -c "import pandas" 2>/dev/null; then
    echo "[INFO] Installing ML driver deps (minimal set, may take a few minutes)..."
    PIP_OPTS="--retries 10 --timeout 120 --default-timeout=120"
    REQ_MIN="cluster/requirements_ml_driver_min.txt"
    REQ_FULL="cluster/requirements_ml_driver.txt"
    if [ -f "\$REQ_MIN" ]; then
        pip install \$PIP_OPTS -r "\$REQ_MIN" || {
            echo "[WARN] Batch install failed — trying one package at a time..."
            pip install \$PIP_OPTS pandas
            pip install \$PIP_OPTS matplotlib
            pip install \$PIP_OPTS seaborn
        }
    elif [ -f "\$REQ_FULL" ]; then
        pip install \$PIP_OPTS -r "\$REQ_FULL"
    else
        pip install \$PIP_OPTS pandas matplotlib seaborn
    fi
fi
python -c "import pandas, matplotlib, seaborn; print('[OK] ML deps ready')"
export JAVA_HOME="\${JAVA_HOME:-\$(dirname "\$(dirname "\$(readlink -f "\$(which java)")")")}"
export PATH="\$JAVA_HOME/bin:\$PATH"
echo "[INFO] JAVA_HOME=\$JAVA_HOME"
export IDS_SPARK_CLUSTER=1
export IDS_ROOT="$REMOTE_ROOT"
export IDS_CLUSTER_DATA_DIR="${IDS_CLUSTER_DATA_DIR:-$REMOTE_ROOT/data}"
export SPARK_MASTER="${SPARK_MASTER}"
export SPARK_DRIVER_HOST="${SPARK_DRIVER_HOST:-$(echo "$DRIVER" | cut -d@ -f2)}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-2g}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-3g}"
export SPARK_EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-4}"
export SPARK_SHUFFLE_PARTITIONS="${SPARK_SHUFFLE_PARTITIONS:-16}"
unset IDS_ALLOW_LOCAL_SPARK
python "$SCRIPT_BASE" "$@"
EOF

echo ""
echo "[OK] Finished: $SCRIPT_BASE"
echo ""
