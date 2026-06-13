#!/bin/bash
# Source cluster/spark_cluster.env — required for all reproduce / ML runs.
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$CLUSTER_DIR/spark_cluster.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "[ERR] Missing cluster/spark_cluster.env"
    echo "      cp cluster/spark_cluster.env.example cluster/spark_cluster.env"
    echo "      Edit MAC_IP, JETSON1_IP, JETSON2_IP, SSH user, paths."
    exit 1
fi

# shellcheck disable=SC1091
source "$ENV_FILE"

export IDS_SPARK_CLUSTER=1

if [[ "${SPARK_MASTER:-}" != spark://* ]]; then
    echo "[ERR] SPARK_MASTER must be spark://<MAC_IP>:7077 in cluster/spark_cluster.env"
    exit 1
fi

if [ -z "${IDS_CLUSTER_DATA_DIR:-}" ] || [ -z "${CLUSTER_DRIVER:-}" ]; then
    echo "[ERR] Set IDS_CLUSTER_DATA_DIR and CLUSTER_DRIVER in cluster/spark_cluster.env"
    exit 1
fi
