#!/bin/bash
# run_all.sh — one-shot distributed run on the Jetson cluster, feeding thesis + FAIR + SOICT.
# Reuses existing infra (cluster/run_ml_remote.sh); resume markers, timing log, error handling.
#
# Prereqs (same as reproduce_cluster.sh): cluster/spark_cluster.env filled, SSH to Jetson, CSVs in data/.
# Usage:
#   ./run_all.sh                        # all phases, auto-resume
#   ./run_all.sh offline | edge         # one phase only
#   FORCE=1 ./run_all.sh                # ignore resume markers
#   STOP_ON_ERROR=0 ./run_all.sh        # keep going on failure, report at end
#   RUN_STREAMING=1 ./run_all.sh edge   # include streaming benchmark (needs Docker+Kafka)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CLUSTER_DIR="$ROOT/cluster"
STATE_DIR="$ROOT/.run_all_state"
LOG_DIR="$ROOT/output/run_all_logs"
TIMING_LOG="$LOG_DIR/timing_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$STATE_DIR" "$LOG_DIR"

PHASE="${1:-all}"                       # all | offline | edge
FORCE="${FORCE:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
RUN_STREAMING="${RUN_STREAMING:-0}"

source "$CLUSTER_DIR/load_cluster_env.sh"
DRIVER="${CLUSTER_DRIVER:?Set CLUSTER_DRIVER in cluster/spark_cluster.env}"
REMOTE_ROOT="${CLUSTER_DRIVER_IDS_ROOT:-/home/jetson/Thesis_IDS}"
SSH_OPTS="${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"

FAILED_STEPS=()
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$TIMING_LOG"; }
hr()  { echo "================================================================" | tee -a "$TIMING_LOG"; }

# run_step <name> <command...> — resume + timing + error handling
run_step() {
  local name="$1"; shift
  local marker="$STATE_DIR/$name.done"
  if [ "$FORCE" != "1" ] && [ -f "$marker" ]; then
    log "SKIP  $name (done)"; return 0
  fi
  hr; log "START $name"; local t0=$(date +%s)
  if "$@" >>"$LOG_DIR/$name.log" 2>&1; then
    local dt=$(( $(date +%s) - t0 )); printf -v h '%02d:%02d:%02d' $((dt/3600)) $((dt%3600/60)) $((dt%60))
    touch "$marker"; log "DONE  $name (${h})"
  else
    FAILED_STEPS+=("$name"); log "FAIL  $name -> output/run_all_logs/$name.log"
    [ "$STOP_ON_ERROR" = "1" ] && { log "Stopped. Re-run ./run_all.sh to resume from here."; exit 1; }
  fi
}

remote()    { "$CLUSTER_DIR/run_ml_remote.sh" "$@"; }
local_mac() {
  cd "$ROOT"; [ -d venv/bin ] && source venv/bin/activate
  IDS_ROOT="$ROOT" IDS_DATA_DIR="${IDS_DATA_DIR:-$ROOT/data}" IDS_ALLOW_LOCAL_SPARK=1 python "$@"
}
ssh_driver() {
  ssh $SSH_OPTS "$DRIVER" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -d jetson/venv/bin ]; then source jetson/venv/bin/activate; elif [ -d venv/bin ]; then source venv/bin/activate; fi
$*
EOF
}
ensure_master() {
  "$CLUSTER_DIR/check_cluster.sh" >/dev/null 2>&1 || { log "Starting Spark master..."; "$CLUSTER_DIR/start_master_mac.sh"; sleep 3; }
}

# Phase 1: offline Spark on the cluster (ml_02/ml_06 must precede ml_07)
phase_offline() {
  hr; log "PHASE 1 — OFFLINE (distributed Spark)"; hr
  ensure_master
  # ml_00 runs on Mac (raw CSVs are Mac-only), then Parquet is synced to workers
  if [ "$FORCE" = "1" ] || [ ! -f "${IDS_DATA_DIR:-$ROOT/data}/train_data.parquet" ]; then
    run_step ml_00_prepare local_mac ml_00_prepare_cicids2017.py
    run_step sync_after_prep "$CLUSTER_DIR/sync_workspace.sh"
  fi
  run_step ml_02_rf_importance  remote ml_02_feature_selection_rf.py
  run_step ml_05_shap_explain   remote ml_05_shap_explainability.py
  run_step ml_06_shap_selection remote ml_06_feature_selection_shap.py
  run_step ml_01_baseline       remote ml_01_baseline_all_features.py
  run_step ml_04_pca            remote ml_04_dimensionality_reduction_pca.py
  run_step ml_03_tuning         remote ml_03_hyperparameter_tuning.py
  run_step ml_07_comparison     remote ml_07_cross_method_comparison.py
  run_step ml_09_multiclass     remote ml_09_multiclass_eval.py
  run_step ml_10_leakage_abl    remote ml_10_leakage_ablation.py
  run_step ml_11_cross_dataset  remote ml_11_cross_dataset_eval.py
  run_step ml_08_anomaly_gate   remote ml_08_anomaly_gate_autoencoder.py
  [ -f "$CLUSTER_DIR/pull_results.sh" ] && run_step pull_results bash "$CLUSTER_DIR/pull_results.sh"
}

# Phase 2: edge benchmarks on Jetson (non-streaming needs no Kafka/Docker)
phase_edge() {
  hr; log "PHASE 2 — EDGE (Jetson inference)"; hr
  run_step save_models local_mac jetson/scripts/save_all_models.py
  run_step sync_models "$CLUSTER_DIR/sync_workspace.sh"
  run_step bench_engines     ssh_driver "python jetson/scripts/benchmark_engines.py"
  run_step bench_all_models  ssh_driver "python jetson/scripts/benchmark_all.py"
  run_step bench_distributed ssh_driver "python jetson/scripts/benchmark_distributed.py"
  if [ "$RUN_STREAMING" = "1" ]; then
    run_step bench_streaming ssh_driver "python jetson/scripts/benchmark_all.py --streaming"
  else
    log "Skipping end-to-end streaming benchmark (needs Docker+Kafka; use RUN_STREAMING=1)."
  fi
}

hr; log "run_all.sh  phase=$PHASE force=$FORCE stop_on_error=$STOP_ON_ERROR"
log "Master=$SPARK_MASTER  Driver=$DRIVER"; hr
case "$PHASE" in
  offline) phase_offline ;;
  edge)    phase_edge ;;
  all)     phase_offline; phase_edge ;;
  *) echo "Usage: $0 {all|offline|edge}"; exit 1 ;;
esac

hr
if [ "${#FAILED_STEPS[@]}" -eq 0 ]; then
  log "[OK] All steps done. Offline -> results/ ; edge -> output/."
else
  log "[FAIL] ${#FAILED_STEPS[@]} step(s): ${FAILED_STEPS[*]} — fix and re-run ./run_all.sh to resume."; exit 1
fi
hr
