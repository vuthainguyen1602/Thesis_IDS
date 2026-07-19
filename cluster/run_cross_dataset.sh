#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cross-dataset evaluation (ml_11): CICIDS2017 <-> CSE-CIC-IDS2018.
# RUN ON THE MAC. One command, resumable:
#
#   ./cluster/run_cross_dataset.sh
#
# Steps (markers in .xd_state/ let you re-run safely):
#   1. prepare  — ml_00 on the 2018 CSVs, LOCAL Spark on the Mac
#                 (IDS_SAMPLE_FRAC=0.2 stratified sample -> ~3.2M flows,
#                 same scale the Mac already handled for 2017)
#   2. sync     — rsync data/ (2017) + data_2018/ parquet to BOTH Jetsons
#                 (cluster executors read file:// paths locally on each node)
#   3. train    — ml_11 on the Spark cluster (driver Jetson#1, 2 workers)
#   4. pull     — copy results/ml_11_cross_dataset_eval back to the Mac
#
# Requires: Spark master on Mac + workers on both Jetsons already running
# for step 3 (./cluster/start_master_mac.sh + ./cluster/start_worker.sh).
# ---------------------------------------------------------------------------
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"
RAW_2018="${RAW_2018:-$HOME/Desktop/ids-2018}"
STATE="$ROOT/.xd_state"; mkdir -p "$STATE"

source "$CLUSTER_DIR/spark_cluster.env"
SSH_OPTS="${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=20}"
REMOTE_ROOT="${CLUSTER_DRIVER_IDS_ROOT:?}"

step() { [ -f "$STATE/$1.done" ] && { echo "[skip] $1 (done)"; return 1; } || { echo ""; echo "===== STEP: $1 ====="; return 0; }; }
mark() { touch "$STATE/$1.done"; }

# ── 1. prepare 2018 locally on the Mac ─────────────────────────────────────
if step prepare_2018; then
  [ -d "$RAW_2018" ] || { echo "[ERR] Raw 2018 CSVs not found at $RAW_2018"; exit 1; }
  cd "$ROOT"
  export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
  export PATH="$JAVA_HOME/bin:$PATH"
  # local Spark only — make sure no cluster vars leak in
  unset IDS_SPARK_CLUSTER SPARK_MASTER SPARK_DRIVER_HOST || true
  export SPARK_DRIVER_MEMORY=8g
  IDS_DATASET=cicids2018 \
  IDS_RAW_DATA_DIR="$RAW_2018" \
  IDS_CSV_GLOB='*.csv' \
  IDS_SAMPLE_FRAC=0.2 \
  IDS_DATA_DIR="$ROOT/data_2018" \
    python ml_00_prepare_cicids2017.py
  mark prepare_2018
fi

# ── 2. sync parquet to BOTH Jetsons ────────────────────────────────────────
if step sync_data; then
  for H in "$JETSON1_IP" "$JETSON2_IP"; do
    echo "[sync] data_2018 + data -> $H"
    rsync -az -e "ssh $SSH_OPTS" "$ROOT/data_2018" "$JETSON_SSH_USER@$H:$REMOTE_ROOT/"
    rsync -az -e "ssh $SSH_OPTS" "$ROOT/data"      "$JETSON_SSH_USER@$H:$REMOTE_ROOT/"
  done
  mark sync_data
fi

# ── 3. ml_11 on the Spark cluster ──────────────────────────────────────────
if step train_ml11; then
  # sanity: master reachable?
  (echo > /dev/tcp/${MAC_IP}/7077) 2>/dev/null || {
    echo "[ERR] Spark master not reachable at ${MAC_IP}:7077."
    echo "      Start it first:  ./cluster/start_master_mac.sh  (Mac)"
    echo "      and on each Jetson:  ./cluster/start_worker.sh"; exit 1; }
  ssh $SSH_OPTS "$CLUSTER_DRIVER" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -d jetson/venv/bin ]; then source jetson/venv/bin/activate; elif [ -d venv/bin ]; then source venv/bin/activate; fi
export JAVA_HOME="\${JAVA_HOME:-\$(dirname "\$(dirname "\$(readlink -f "\$(which java)")")")}"
export PATH="\$JAVA_HOME/bin:\$PATH"
export IDS_SPARK_CLUSTER=1
export IDS_ROOT="$REMOTE_ROOT"
export SPARK_MASTER="$SPARK_MASTER"
export SPARK_DRIVER_HOST="$JETSON1_IP"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-3g}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-3g}"
export SPARK_DRIVER_MAX_RESULT_SIZE="${SPARK_DRIVER_MAX_RESULT_SIZE:-3g}"
export SPARK_EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-4}"
export SPARK_SHUFFLE_PARTITIONS="${SPARK_SHUFFLE_PARTITIONS:-32}"
export IDS_XD_DIR_A="$REMOTE_ROOT/data"       IDS_XD_NAME_A=CICIDS2017
export IDS_XD_DIR_B="$REMOTE_ROOT/data_2018"  IDS_XD_NAME_B=CSE-CIC-IDS2018
python ml_11_cross_dataset_eval.py
EOF
  mark train_ml11
fi

# ── 4. pull results back ───────────────────────────────────────────────────
if step pull_results; then
  mkdir -p "$ROOT/results"
  rsync -az -e "ssh $SSH_OPTS" \
    "$CLUSTER_DRIVER:$REMOTE_ROOT/results/ml_11_cross_dataset_eval" "$ROOT/results/"
  mark pull_results
  echo ""
  echo "[DONE] Results:"
  ls -la "$ROOT/results/ml_11_cross_dataset_eval/"
  echo "Now fill tab:cross-dataset from cross_dataset_results.csv"
fi
