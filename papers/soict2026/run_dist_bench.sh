#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-mode distributed-benchmark orchestrator for the SOICT edge table and the
# journal extension. RUN THIS ON THE MAC. It manages both Jetson pipelines over
# SSH (key auth).
#
#   ./run_dist_bench.sh single              # 1 board, full pipeline (Jetson #2), gate OFF
#   ./run_dist_bench.sh single_gate         # 1 board, full pipeline with the gate inline
#   ./run_dist_bench.sh split               # gate on #1 + classifier on #2
#   ./run_dist_bench.sh split_1board        # gate + classifier as TWO processes on #2
#   ./run_dist_bench.sh horizontal          # both boards full, same Kafka group, gate OFF
#   ./run_dist_bench.sh horizontal_gate     # both boards full, gate inline
#   ./run_dist_bench.sh spark_cluster       # Mac=master, both Jetsons=workers
#   ./run_dist_bench.sh <mode>_onnx         # same placement, classifier on ONNX Runtime:
#                                           #   single_onnx single_gate_onnx split_1board_onnx
#                                           #   split_onnx horizontal_onnx horizontal_gate_onnx
#   ./run_dist_bench.sh merge               # build summary.csv from all JSON
#
# Do ONE mode at a time, watch the output, then the next. Tunables (env):
# RATE=100 DUR=60 WARM=30 REP=5 COOL=120 NO_ENERGY=1 OUT_DIR=<results dir>.
# Each mode records the board clocks' offset against the Mac and the ping RTT
# before and after (clock_<mode>_<T>_{before,after}.json); analyze_runs.py
# takes those off the end-to-end latency with --clock-json.
# NOTE: run-level p95 here comes from the InfluxDB load-window (the code prints
# a caveat). For raw-CSV p95 the raw logs are synced to the results dir.
# ---------------------------------------------------------------------------
set -uo pipefail

# ---- cluster config (from cluster/spark_cluster.env) ----
MAC_IP=${MAC_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)}  # the Mac's DHCP address changes
J1=${J1:-192.168.1.81}    # jetson-nano-1 (was 192.168.1.50 until 2026-10)
J2=${J2:-192.168.1.39}     # jetson-nano-2 (was 192.168.1.204 until 2026-10)
# known_hosts still files the boards' keys under their old IPs
J1_KEY=192.168.1.50; J2_KEY=192.168.1.204
hka() { case $1 in "$J1") echo "-o HostKeyAlias=$J1_KEY" ;; "$J2") echo "-o HostKeyAlias=$J2_KEY" ;; esac; }
U=bvdung
JROOT=/home/$U/Thesis_IDS/jetson
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
# benchmark_distributed.py snapshots these boards' clocks around every load window.
export CLOCK_TARGETS="$U@$J1:$J1_KEY $U@$J2:$J2_KEY"

# ---- tunables (fast draft defaults; raise REP/DUR for the final numbers) ----
RATE=${RATE:-100}; DUR=${DUR:-45}; WARM=${WARM:-20}; REP=${REP:-3}
# Pause between repeats; long enough for a backlogged classifier queue to drain.
COOL=${COOL:-120}

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
MACJ="$ROOT/jetson"
OUT="${OUT_DIR:-$DIR/results/benchmarks}"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"  # absolute: python steps cd elsewhere
CSV_ABS="$MACJ/sender/replay_cicids2017.csv"
ts() { date +%Y%m%d_%H%M%S; }

# Mac-side orchestrator env: point config.py at the Mac's Docker services and
# the replay CSV. Exported vars win over jetson/.env (load_dotenv no-override).
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export POSTGRES_HOST=localhost POSTGRES_PORT=5433
export INFLUXDB_URL=http://localhost:8086
export DATA_CSV_PATH="$CSV_ABS"

# Inline env every remote pipeline gets (wins over stale .env values/IPs).
# A fresh consumer group per run: with a shared group, flows a stopped run had not
# consumed stay committed-but-unread in Kafka and the next run would process them.
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
REMOTE_ENV="KAFKA_BOOTSTRAP_SERVERS=$MAC_IP:9092 POSTGRES_HOST=$MAC_IP POSTGRES_PORT=5433 INFLUXDB_URL=http://$MAC_IP:8086 KAFKA_GROUP_ID=ids-bench-$RUN_TAG KAFKA_CLASSIFIER_GROUP_ID=ids-bench-clf-$RUN_TAG"

ensure_mac_deps() {
  python - <<'PY' 2>/dev/null && return 0 || true
import kafka, psycopg2, influxdb_client, psutil, pandas  # noqa
PY
  echo "[deps] installing Mac orchestrator deps ..."
  pip install -q kafka-python psycopg2-binary influxdb-client psutil pandas || \
    pip install -q --break-system-packages kafka-python psycopg2-binary influxdb-client psutil pandas
}

stop_all() {
  echo "[stop] killing consumers + power monitors on both Jetsons"
  for H in $J1 $J2; do
    $SSH $(hka $H) $U@$H "pkill -f kafka_consumer.py 2>/dev/null; pkill -f 'benchmark_distributed.py node-power' 2>/dev/null; true"
  done
  sleep 3
}

# clock_snapshot T label  -> $OUT/clock_<mode>_<T>_<label>.json (both boards)
clock_snapshot() {
  local T=$1 label=$2
  echo "[clock] board clock offsets + ping RTT ($label)"
  local f="$OUT/clock_${MODE_NAME}_${T}_${label}.json"
  ( cd "$MACJ" && python scripts/clock_offset.py -n 40 --ping 20 --out "$f" \
      "$U@$J1:$J1_KEY" "$U@$J2:$J2_KEY" >/dev/null )
  python - "$f" <<'PY'
import json, sys
for h in json.load(open(sys.argv[1]))["hosts"]:
    print("  %s: offset %+.4f s (+/-%.1f ms)  ping %s ms avg / %s max, loss %s%%" % (
        h["host"], h["offset_s"], h["error_bound_s"] * 1000,
        h.get("ping_rtt_avg_ms", "?"), h.get("ping_rtt_max_ms", "?"), h.get("ping_loss_pct", "?")))
PY
}

# start_pipe host node_id role alert [spark_master] [driver_host]
# EXTRA_REMOTE_ENV (e.g. "ANOMALY_ENABLED=1 EDGE_ENGINE=onnx") is appended.
start_pipe() {
  local host=$1 id=$2 role=$3 alert=$4 sm=${5-} dh=${6-}
  local spark="SPARK_MASTER="                         # empty -> local[*]
  [ -n "$sm" ] && spark="SPARK_MASTER=$sm SPARK_DRIVER_HOST=$dh"
  echo "[start] $id role=$role alert=$alert spark=${sm:-local[*]} extra='${EXTRA_REMOTE_ENV:-}' on $host"
  # Background the ssh CLIENT locally so a lingering channel can never block us;
  # the remote pipeline is nohup+detached so it survives the ssh client exiting.
  $SSH $(hka $host) -n $U@$host "cd $JROOT && source venv/bin/activate && \
    $REMOTE_ENV ${EXTRA_REMOTE_ENV:-} $spark RAW_LATENCY_LOG=\$HOME/ids_raw_latency_${id}.csv \
    EDGE_NODE_ID=$id EDGE_NODE_ROLE=$role ALERT_ENABLED=$alert \
    nohup python edge/kafka_consumer.py > \$HOME/edge_${id}.log 2>&1 </dev/null &" &
  sleep 5
  echo "  -> launch dispatched on $host (log: ~/edge_${id}.log)"
}

# energy_window ts spec...   spec = host:node_id:role
energy_window() {
  local T=$1; shift
  local pd=$((DUR + 30))          # sampling window must cover the DUR-second load
  echo "[energy] node-power on: $* (idle 30s + sample ${pd}s)"
  for spec in "$@"; do IFS=: read -r host id role <<<"$spec"
    $SSH $(hka $host) -n $U@$host "cd $JROOT && source venv/bin/activate && \
      EDGE_NODE_ID=$id EDGE_NODE_ROLE=$role nohup python scripts/benchmark_distributed.py node-power \
      --duration $pd --idle-seconds 30 --output \$HOME/power_${id}_${T}.json \
      > \$HOME/power_${id}.log 2>&1 </dev/null &" &
    sleep 3; echo "  -> power monitor dispatched on $host"
  done
  echo "[energy] idle baseline in progress — NOT sending load for 35s ..."; sleep 35
  echo "[energy] sending ${DUR}s load for the energy window ..."
  ( cd "$MACJ" && python scripts/benchmark_distributed.py send \
      --duration "$DUR" --rate "$RATE" --output "$OUT/send_energy_${T}.json" ) || true
  # node-power lifetime = 30s idle + pd sample; we are ~35+DUR s in, wait the rest.
  echo "[energy] waiting for power monitors to finish (~$((pd - DUR + 10))s) ..."; sleep $((pd - DUR + 10))
}

sync_logs() {  # T  active-hosts...
  local T=$1; shift
  for H in "$@"; do
    echo "[sync] pulling raw latency + power JSON from $H"
    scp -o StrictHostKeyChecking=accept-new $(hka $H) "$U@$H:~/ids_raw_latency_*.csv" "$OUT/" 2>/dev/null || true
    scp -o StrictHostKeyChecking=accept-new $(hka $H) "$U@$H:~/power_*_${T}.json"      "$OUT/" 2>/dev/null || true
  done
}

# Which boards a mode uses (for the energy window and log sync).
mode_boards() {
  case "$1" in
    single|single_gate|single_onnx|single_gate_onnx|split_1board|split_1board_onnx) echo "J2" ;;
    split|split_onnx|spark_cluster|horizontal|horizontal_gate|horizontal_onnx|horizontal_gate_onnx) echo "J1 J2" ;;
    *) echo "" ;;
  esac
}

run_mode() {
  local mode=$1 T; T=$(ts); MODE_NAME=$mode
  ensure_mac_deps
  stop_all
  clock_snapshot "$T" before
  local ONNX=""; case "$mode" in *_onnx) ONNX="EDGE_ENGINE=onnx" ;; esac
  case "$mode" in
    single|single_onnx)
                 EXTRA_REMOTE_ENV="$ONNX" start_pipe $J2 jetson-nano-2 full 1 ;;
    single_gate|single_gate_onnx)
                 EXTRA_REMOTE_ENV="ANOMALY_ENABLED=1 $ONNX" start_pipe $J2 jetson-nano-2 full 1 ;;
    split|split_onnx)
                 start_pipe $J1 jetson-nano-1 anomaly_gate 0
                 EXTRA_REMOTE_ENV="$ONNX" start_pipe $J2 jetson-nano-2 classifier 1 ;;
    split_1board|split_1board_onnx)
                 # Same pipeline split as `split`, but both tiers on Jetson #2 as
                 # two processes; the node ids keep their metrics and raw logs apart.
                 start_pipe $J2 jetson-nano-2-gate anomaly_gate 0
                 EXTRA_REMOTE_ENV="$ONNX" start_pipe $J2 jetson-nano-2-clf classifier 1 ;;
    horizontal|horizontal_onnx)
                 EXTRA_REMOTE_ENV="$ONNX" start_pipe $J1 jetson-nano-1 full 1
                 EXTRA_REMOTE_ENV="$ONNX" start_pipe $J2 jetson-nano-2 full 0 ;;
    horizontal_gate|horizontal_gate_onnx)
                 EXTRA_REMOTE_ENV="ANOMALY_ENABLED=1 $ONNX" start_pipe $J1 jetson-nano-1 full 1
                 EXTRA_REMOTE_ENV="ANOMALY_ENABLED=1 $ONNX" start_pipe $J2 jetson-nano-2 full 0 ;;
    spark_cluster)
                 echo "[spark] start master on Mac + workers on Jetsons FIRST:"
                 echo "        (Mac)     ./cluster/start_master_mac.sh"
                 echo "        (each J)  ./cluster/start_worker.sh"
                 read -r -p "Press ENTER once master+workers are up... " _
                 start_pipe $J2 jetson-nano-2 classifier 1 "spark://$MAC_IP:7077" "$J2"
                 start_pipe $J1 jetson-nano-1 anomaly_gate 0 ;;
    *) echo "unknown mode: $mode"; exit 1 ;;
  esac
  case "$mode" in
    *_onnx) echo "[wait] letting pipelines initialise (ONNX, no Spark) ..."; sleep 15 ;;
    *)      echo "[wait] letting pipelines initialise (Spark ~15s) ..."; sleep 25 ;;
  esac

  # 1) throughput / latency / F1  (run sends its own warmup+load, REP repeats)
  echo "[run] throughput/latency, mode=$mode, repeats=$REP ..."
  ( cd "$MACJ" && python scripts/benchmark_distributed.py run \
      --mode "$mode" --duration "$DUR" --rate "$RATE" --warmup "$WARM" --repeats "$REP" --cooldown "$COOL" \
      --output "$OUT/run_${mode}_${T}.json" )

  # 2) energy window (node-power on the active node(s)); NO_ENERGY=1 skips it for
  #    rate sweeps, where only throughput and latency are needed.
  if [ "${NO_ENERGY:-0}" = "1" ]; then
    clock_snapshot "$T" after; stop_all; echo "[done] mode=$mode (no energy window)"; return
  fi
  case "$mode" in
    single|single_gate|single_onnx|single_gate_onnx|split_1board|split_1board_onnx)
                   energy_window "$T" "$J2:jetson-nano-2:full" ; ACTIVE="$J2" ;;
    split|split_onnx|spark_cluster)
                   energy_window "$T" "$J1:jetson-nano-1:anomaly_gate" "$J2:jetson-nano-2:classifier"; ACTIVE="$J1 $J2" ;;
    horizontal|horizontal_gate|horizontal_onnx|horizontal_gate_onnx)
                   energy_window "$T" "$J1:jetson-nano-1:full" "$J2:jetson-nano-2:full"; ACTIVE="$J1 $J2" ;;
  esac

  sync_logs "$T" $ACTIVE
  clock_snapshot "$T" after
  stop_all
  echo "[done] mode=$mode  ->  $OUT/run_${mode}_${T}.json (+ power_*_${T}.json)"
}

case "${1:-}" in
  single|single_gate|split|split_1board|horizontal|horizontal_gate|spark_cluster|\
  single_onnx|single_gate_onnx|split_1board_onnx|split_onnx|horizontal_onnx|horizontal_gate_onnx)
    run_mode "$1" ;;
  merge)
    ensure_mac_deps
    ( cd "$MACJ" && python scripts/benchmark_distributed.py merge \
        --input "$OUT/*.json" --output-csv "$OUT/summary.csv" )
    echo "[merge] -> $OUT/summary.csv" ;;
  *)
    echo "Usage: $0 {single|single_gate|split|split_1board|horizontal|horizontal_gate|spark_cluster|<mode>_onnx|merge}"; exit 1 ;;
esac
