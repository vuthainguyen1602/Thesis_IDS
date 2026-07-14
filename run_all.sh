#!/bin/bash
# =============================================================================
#  run_all.sh — Chạy MỘT phát toàn bộ thí nghiệm trên cụm Jetson phân tán,
#  sinh dữ liệu cho CẢ luận văn + FAIR + SOICT.
#
#  Thiết kế: tái dùng hạ tầng sẵn có (cluster/run_ml_remote.sh lo env/deps/ssh),
#  KHÔNG thêm setup mới. Có RESUME (chạy lại là bỏ qua bước đã xong), ghi
#  THỜI GIAN từng bước, và BẮT LỖI rõ ràng cho lần chạy dài ~1–2 ngày.
#
#  Chuẩn bị 1 lần (giống reproduce_cluster.sh — không thêm gì):
#    - Điền cluster/spark_cluster.env (đã có .example)
#    - SSH tới Jetson chạy được (khóa công khai)
#    - Có bộ CSV CICIDS2017 trong data/ trên Mac
#
#  Dùng:
#    ./run_all.sh                 # chạy tất cả (offline + biên), tự resume
#    ./run_all.sh offline         # chỉ pha offline (Spark trên cụm)
#    ./run_all.sh edge            # chỉ pha biên (benchmark Jetson)
#    FORCE=1 ./run_all.sh         # chạy lại từ đầu, bỏ qua marker resume
#    STOP_ON_ERROR=0 ./run_all.sh # gặp lỗi vẫn chạy tiếp, tổng kết cuối
#    RUN_STREAMING=1 ./run_all.sh edge   # chạy cả benchmark streaming (cần Docker+Kafka)
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CLUSTER_DIR="$ROOT/cluster"
STATE_DIR="$ROOT/.run_all_state"
LOG_DIR="$ROOT/output/run_all_logs"
TIMING_LOG="$LOG_DIR/timing_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$STATE_DIR" "$LOG_DIR"

PHASE="${1:-all}"                       # all | offline | edge
FORCE="${FORCE:-0}"                     # 1 = bỏ qua marker resume
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"     # 1 = dừng khi bước lỗi
RUN_STREAMING="${RUN_STREAMING:-0}"     # 1 = chạy benchmark streaming (cần Docker+Kafka)

source "$CLUSTER_DIR/load_cluster_env.sh"
DRIVER="${CLUSTER_DRIVER:?Điền CLUSTER_DRIVER trong cluster/spark_cluster.env}"
REMOTE_ROOT="${CLUSTER_DRIVER_IDS_ROOT:-/home/jetson/Thesis_IDS}"
SSH_OPTS="${CLUSTER_SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"

FAILED_STEPS=()

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$TIMING_LOG"; }
hr()   { echo "================================================================" | tee -a "$TIMING_LOG"; }

# run_step <name> <command...>  — có resume + timing + bắt lỗi
run_step() {
  local name="$1"; shift
  local marker="$STATE_DIR/$name.done"
  if [ "$FORCE" != "1" ] && [ -f "$marker" ]; then
    log "SKIP  $name (đã xong — xóa $marker hoặc FORCE=1 để chạy lại)"
    return 0
  fi
  hr; log "BẮT ĐẦU  $name"; local t0=$(date +%s)
  if "$@" >>"$LOG_DIR/$name.log" 2>&1; then
    local dt=$(( $(date +%s) - t0 ))
    printf -v h '%02d:%02d:%02d' $((dt/3600)) $((dt%3600/60)) $((dt%60))
    touch "$marker"; log "XONG   $name  (${h})  → log: output/run_all_logs/$name.log"
  else
    local dt=$(( $(date +%s) - t0 )); FAILED_STEPS+=("$name")
    log "LỖI    $name  (sau ${dt}s)  → xem output/run_all_logs/$name.log"
    if [ "$STOP_ON_ERROR" = "1" ]; then
      log "Dừng lại. Sửa xong chạy lại ./run_all.sh — sẽ tự tiếp tục từ bước này."
      exit 1
    fi
  fi
}

# --- helper chạy 1 script ML trên cụm (qua hạ tầng sẵn có) ---
remote()    { "$CLUSTER_DIR/run_ml_remote.sh" "$@"; }
# --- helper chạy 1 script trên Mac ở chế độ Spark local (cho ml_00: CSV thô nằm ở Mac) ---
local_mac() {
  cd "$ROOT"
  [ -d venv/bin ] && source venv/bin/activate
  IDS_ROOT="$ROOT" IDS_DATA_DIR="${IDS_DATA_DIR:-$ROOT/data}" IDS_ALLOW_LOCAL_SPARK=1 \
    python "$@"
}
# --- helper ssh chạy lệnh tùy ý trên Jetson driver (cho benchmark biên) ---
ssh_driver() {
  ssh $SSH_OPTS "$DRIVER" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -d jetson/venv/bin ]; then source jetson/venv/bin/activate; elif [ -d venv/bin ]; then source venv/bin/activate; fi
$*
EOF
}

ensure_master() {
  "$CLUSTER_DIR/check_cluster.sh" >/dev/null 2>&1 || {
    log "Khởi động Spark master trên Mac..."; "$CLUSTER_DIR/start_master_mac.sh"; sleep 3;
  }
}

# =============================== PHA 1: OFFLINE ===============================
phase_offline() {
  hr; log "PHA 1 — OFFLINE (Spark phân tán trên cụm Jetson)"; hr
  ensure_master

  # ml_00 chạy trên Mac (CSV thô chỉ có ở Mac), rồi đồng bộ Parquet sang worker
  if [ "$FORCE" = "1" ] || [ ! -f "${IDS_DATA_DIR:-$ROOT/data}/train_data.parquet" ]; then
    run_step ml_00_prepare local_mac ml_00_prepare_cicids2017.py
    run_step sync_after_prep "$CLUSTER_DIR/sync_workspace.sh"
  else
    log "SKIP  ml_00_prepare (đã có train_data.parquet)"
  fi

  # Thứ tự có phụ thuộc: ml_02 (RF importance) và ml_06 (SHAP importance) TRƯỚC ml_07
  run_step ml_02_rf_importance   remote ml_02_feature_selection_rf.py
  run_step ml_05_shap_explain    remote ml_05_shap_explainability.py
  run_step ml_06_shap_selection  remote ml_06_feature_selection_shap.py
  run_step ml_01_baseline        remote ml_01_baseline_all_features.py
  run_step ml_04_pca             remote ml_04_dimensionality_reduction_pca.py
  run_step ml_03_tuning          remote ml_03_hyperparameter_tuning.py
  run_step ml_07_comparison      remote ml_07_cross_method_comparison.py   # THÍ NGHIỆM CHÍNH
  run_step ml_09_multiclass      remote ml_09_multiclass_eval.py
  run_step ml_10_leakage_abl     remote ml_10_leakage_ablation.py
  run_step ml_11_cross_dataset   remote ml_11_cross_dataset_eval.py
  run_step ml_08_anomaly_gate    remote ml_08_anomaly_gate_autoencoder.py

  # Kéo kết quả/CSV/hình từ driver về Mac (nếu script này có sẵn)
  [ -f "$CLUSTER_DIR/pull_results.sh" ] && run_step pull_results bash "$CLUSTER_DIR/pull_results.sh"
}

# =============================== PHA 2: BIÊN =================================
phase_edge() {
  hr; log "PHA 2 — BIÊN (suy luận/benchmark trên Jetson)"; hr
  # Lưu model đã huấn luyện (artifact độc lập phần cứng) rồi đồng bộ sang Jetson
  run_step save_models  local_mac jetson/scripts/save_all_models.py
  run_step sync_models  "$CLUSTER_DIR/sync_workspace.sh"

  # Benchmark KHÔNG cần Kafka/Docker (ít setup nhất): tải model + dữ liệu tổng hợp/replay
  run_step bench_engines      ssh_driver "python jetson/scripts/benchmark_engines.py"      # sklearn/ONNX vs Spark
  run_step bench_all_models   ssh_driver "python jetson/scripts/benchmark_all.py"          # độ trễ/thông lượng từng model
  run_step bench_distributed  ssh_driver "python jetson/scripts/benchmark_distributed.py"  # so sánh chế độ phân tán

  if [ "$RUN_STREAMING" = "1" ]; then
    run_step bench_streaming  ssh_driver "python jetson/scripts/benchmark_all.py --streaming"
  else
    log "BỎ QUA benchmark streaming end-to-end (cần Docker+Kafka + edge pipelines chạy)."
    log "  Muốn chạy: khởi động Docker stack + gate/classifier trên Jetson rồi: RUN_STREAMING=1 ./run_all.sh edge"
  fi
}

# ================================= ĐIỀU PHỐI ================================
hr; log "run_all.sh — phase=$PHASE  force=$FORCE  stop_on_error=$STOP_ON_ERROR"
log "Master=$SPARK_MASTER  Driver=$DRIVER"; hr

case "$PHASE" in
  offline) phase_offline ;;
  edge)    phase_edge ;;
  all)     phase_offline; phase_edge ;;
  *) echo "Dùng: $0 {all|offline|edge}"; exit 1 ;;
esac

hr; log "TỔNG KẾT"
if [ "${#FAILED_STEPS[@]}" -eq 0 ]; then
  log "[OK] Tat ca buoc hoan tat. Ket qua: results/ + output/ ; hinh cho papers/thesis da sinh."
  log "  -> FAIR + bang chat luong luan van: results/ml_07*, ml_09*, ml_10*, ml_11*, ml_02/05/06*"
  log "  -> SOICT + bang bien luan van:      output/ (benchmark_*.csv, summary.csv)"
else
  log "[FAIL] Co ${#FAILED_STEPS[@]} buoc LOI: ${FAILED_STEPS[*]}"
  log "  Sua roi chay lai ./run_all.sh (tu resume cac buoc con lai)."
  exit 1
fi
hr
