#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cross-dataset evaluation (ml_11): CICIDS2017 <-> CSE-CIC-IDS2018.
# RUN ON THE MAC. One command, resumable:
#
#   ./cluster/run_cross_dataset.sh
#
# Steps (markers in .xd_state/ let you re-run safely):
#   1a. sync raw 2018 CSVs Mac -> Jetson#1 (NVMe; Mac disk is tight)
#   1b. prepare — ml_00 on Jetson#1 (local Spark, IDS_SAMPLE_FRAC=0.2,
#                 shuffle spills go to the Jetson NVMe, not the Mac disk)
#   2. sync     — copy data_2018 parquet Jetson#1 -> Jetson#2
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

# ── 1a. raw 2018 CSVs -> Jetson#1 (NVMe has plenty of room; the full-data
#        exact-dedup shuffle would exhaust the Mac's remaining disk) ────────
if step sync_raw_2018; then
  [ -d "$RAW_2018" ] || { echo "[ERR] Raw 2018 CSVs not found at $RAW_2018"; exit 1; }
  echo "[sync] raw 2018 CSVs -> $JETSON1_IP:~/ids-2018-raw (~6.5GB, first time only)"
  rsync -az --progress -e "ssh $SSH_OPTS" "$RAW_2018/" \
    "$JETSON_SSH_USER@$JETSON1_IP:/home/$JETSON_SSH_USER/ids-2018-raw/"
  mark sync_raw_2018
fi

# ── 1b. prepare 2018 ON Jetson#1 (local Spark, shuffle spills to NVMe) ─────
if step prepare_2018; then
  ssh $SSH_OPTS "$CLUSTER_DRIVER" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
df -h . | tail -1
if [ -d jetson/venv/bin ]; then source jetson/venv/bin/activate; elif [ -d venv/bin ]; then source venv/bin/activate; fi
export JAVA_HOME="\${JAVA_HOME:-\$(dirname "\$(dirname "\$(readlink -f "\$(which java)")")")}"
export PATH="\$JAVA_HOME/bin:\$PATH"
unset IDS_SPARK_CLUSTER SPARK_MASTER SPARK_DRIVER_HOST || true
export SPARK_DRIVER_MEMORY=6g
IDS_DATASET=cicids2018 \
IDS_RAW_DATA_DIR="/home/$JETSON_SSH_USER/ids-2018-raw" \
IDS_CSV_GLOB='*.csv' \
IDS_SAMPLE_FRAC=0.2 \
IDS_DATA_DIR="$REMOTE_ROOT/data_2018" \
  python ml_00_prepare_cicids2017.py
EOF
  mark prepare_2018
fi

# ── 2. copy the prepared parquet Jetson#1 -> Jetson#2 (executors read the
#       same file:// paths on every node) ───────────────────────────────────
if step sync_data; then
  # J1 has no SSH key for J2 — relay through the Mac (parquet is only a few
  # hundred MB, and we get a local copy of data_2018 as a bonus).
  echo "[sync] data_2018 J1 -> Mac"
  rsync -az --progress -e "ssh $SSH_OPTS" \
    "$CLUSTER_DRIVER:$REMOTE_ROOT/data_2018" "$ROOT/"
  echo "[sync] data_2018 Mac -> J2"
  rsync -az --progress -e "ssh $SSH_OPTS" \
    "$ROOT/data_2018" "$JETSON_SSH_USER@$JETSON2_IP:$REMOTE_ROOT/"
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
# RF 200x15 + dense StandardScaler on ~1.9M rows OOMs 3g/4-core executors
# (exit 52). 4g x 2 cores: same model, half the concurrent task memory pressure.
export SPARK_EXECUTOR_MEMORY="${XD_EXEC_MEM:-4g}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-3g}"
export SPARK_DRIVER_MAX_RESULT_SIZE="${SPARK_DRIVER_MAX_RESULT_SIZE:-3g}"
export SPARK_EXECUTOR_CORES="${XD_EXEC_CORES:-1}"
export SPARK_SHUFFLE_PARTITIONS="${SPARK_SHUFFLE_PARTITIONS:-32}"
export IDS_XD_MAX_MEMORY_MB="${XD_MAXMEM_MB:-128}"
export IDS_XD_NUM_TREES="${XD_TREES:-200}"  IDS_XD_MAX_DEPTH="${XD_DEPTH:-15}"
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
