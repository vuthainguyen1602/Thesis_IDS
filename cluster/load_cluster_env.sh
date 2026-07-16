#!/bin/bash
# Source cluster/spark_cluster.env — required for all reproduce / ML runs.
# Works when sourced from bash or zsh: source cluster/load_cluster_env.sh
set -euo pipefail

if [ -n "${BASH_VERSION:-}" ]; then
    _cluster_script="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _cluster_script="${(%):-%x}"
else
    _cluster_script="$0"
fi
CLUSTER_DIR="$(cd "$(dirname "$_cluster_script")" && pwd)"
ENV_FILE="$CLUSTER_DIR/spark_cluster.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "[ERR] Missing cluster/spark_cluster.env"
    echo "      cp cluster/spark_cluster.env.example cluster/spark_cluster.env"
    echo "      Edit MAC_IP, JETSON1_IP, JETSON2_IP, SSH user, paths."
    exit 1
fi

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

# Auto-trust new host keys + keep long-running SSH sessions alive through brief
# network hiccups (send a keepalive every 30s, tolerate ~10 min unresponsive
# before dropping) so multi-hour remote steps don't die on a WiFi blip.
export CLUSTER_SSH_OPTS="${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -o TCPKeepAlive=yes}"
