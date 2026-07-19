#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-mode distributed-benchmark orchestrator for the SOICT edge table.
# RUN THIS ON THE MAC. It manages both Jetson pipelines over SSH (key auth).
#
#   ./run_dist_bench.sh single         # 1-node full pipeline (Jetson #2)
#   ./run_dist_bench.sh split          # gate on #1 + classifier on #2
#   ./run_dist_bench.sh horizontal     # both nodes full, same Kafka group
#   ./run_dist_bench.sh spark_cluster  # Mac=master, both Jetsons=workers
#   ./run_dist_bench.sh merge          # build summary.csv from all JSON
#
# Do ONE mode at a time, watch the output, then the next. After all four,
# run 'merge'. Tunables (env): RATE=100 DUR=60 WARM=30 REP=5.
# NOTE: run-level p95 here comes from the InfluxDB load-window (the code prints
# a caveat). For raw-CSV p95 the raw logs are synced to results/benchmarks/.
# ---------------------------------------------------------------------------
set -uo pipefail

# ---- cluster config (from cluster/spark_cluster.env) ----
MAC_IP=192.168.1.68
J1=192.168.1.50            # jetson-nano-1
J2=192.168.1.205           # jetson-nano-2
U=bvdung
JROOT=/home/$U/Thesis_IDS/jetson
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=10"

# ---- tunables (fast draft defaults; raise REP/DUR for the final numbers) ----
RATE=${RATE:-100}; DUR=${DUR:-45}; WARM=${WARM:-20}; REP=${REP:-3}

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
MACJ="$ROOT/jetson"
OUT="$DIR/results/benchmarks"; mkdir -p "$OUT"
CSV_ABS="$MACJ/sender/replay_cicids2017.csv"
ts() { date +%Y%m%d_%H%M%S; }

# Mac-side orchestrator env: point config.py at the Mac's Docker services and
# the replay CSV. Exported vars win over jetson/.env (load_dotenv no-override).
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export POSTGRES_HOST=localhost POSTGRES_PORT=5433
export INFLUXDB_URL=http://localhost:8086
export DATA_CSV_PATH="$CSV_ABS"

# Inline env every remote pipeline gets (wins over stale .env values/IPs).
REMOTE_ENV="KAFKA_BOOTSTRAP_SERVERS=$MAC_IP:9092 POSTGRES_HOST=$MAC_IP POSTGRES_PORT=5433 INFLUXDB_URL=http://$MAC_IP:8086 KAFKA_GROUP_ID=ids-jetson-cluster"

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
    $SSH $U@$H "pkill -f kafka_consumer.py 2>/dev/null; pkill -f 'benchmark_distributed.py node-power' 2>/dev/null; true"
  done
  sleep 3
}

# start_pipe host node_id role alert [spark_master] [driver_host]
start_pipe() {
  local host=$1 id=$2 role=$3 alert=$4 sm=${5-} dh=${6-}
  local spark="SPARK_MASTER="                         # empty -> local[*]
  [ -n "$sm" ] && spark="SPARK_MASTER=$sm SPARK_DRIVER_HOST=$dh"
  echo "[start] $id role=$role alert=$alert spark=${sm:-local[*]} on $host"
  $SSH -n $U@$host "cd $JROOT && source venv/bin/activate && \
    $REMOTE_ENV $spark RAW_LATENCY_LOG=\$HOME/ids_raw_latency_\$(hostname).csv \
    EDGE_NODE_ID=$id EDGE_NODE_ROLE=$role ALERT_ENABLED=$alert \
    nohup python edge/kafka_consumer.py > \$HOME/edge_${id}.log 2>&1 </dev/null & echo '  -> launched (pid '\$!')'"
}

# energy_window ts spec...   spec = host:node_id:role
energy_window() {
  local T=$1; shift
  local pd=$((WARM + DUR + 40))
  echo "[energy] node-power on: $* (idle 30s + sample ${pd}s)"
  for spec in "$@"; do IFS=: read -r host id role <<<"$spec"
    $SSH -n $U@$host "cd $JROOT && source venv/bin/activate && \
      EDGE_NODE_ID=$id EDGE_NODE_ROLE=$role nohup python scripts/benchmark_distributed.py node-power \
      --duration $pd --idle-seconds 30 --output \$HOME/power_${id}_${T}.json \
      > \$HOME/power_${id}.log 2>&1 </dev/null & echo '  -> power monitor up'"
  done
  echo "[energy] idle baseline in progress — NOT sending load for 35s ..."; sleep 35
  echo "[energy] sending ${DUR}s load for the energy window ..."
  ( cd "$MACJ" && python scripts/benchmark_distributed.py send \
      --duration "$DUR" --rate "$RATE" --output "$OUT/send_energy_${T}.json" ) || true
  echo "[energy] waiting for power monitors to finish ..."; sleep $((pd - DUR - 30 + 10))
}

sync_logs() {  # T  active-hosts...
  local T=$1; shift
  for H in "$@"; do
    echo "[sync] pulling raw latency + power JSON from $H"
    scp -o StrictHostKeyChecking=accept-new "$U@$H:~/ids_raw_latency_*.csv" "$OUT/" 2>/dev/null || true
    scp -o StrictHostKeyChecking=accept-new "$U@$H:~/power_*_${T}.json"      "$OUT/" 2>/dev/null || true
  done
}

run_mode() {
  local mode=$1 T; T=$(ts)
  ensure_mac_deps
  stop_all
  case "$mode" in
    single)      start_pipe $J2 jetson-nano-2 full 1 ;;
    split)       start_pipe $J1 jetson-nano-1 anomaly_gate 0
                 start_pipe $J2 jetson-nano-2 classifier   1 ;;
    horizontal)  start_pipe $J1 jetson-nano-1 full 1
                 start_pipe $J2 jetson-nano-2 full 0 ;;
    spark_cluster)
                 echo "[spark] start master on Mac + workers on Jetsons FIRST:"
                 echo "        (Mac)     ./cluster/start_master_mac.sh"
                 echo "        (each J)  ./cluster/start_worker.sh"
                 read -r -p "Press ENTER once master+workers are up... " _
                 start_pipe $J2 jetson-nano-2 classifier 1 "spark://$MAC_IP:7077" "$J2"
                 start_pipe $J1 jetson-nano-1 anomaly_gate 0 ;;
    *) echo "unknown mode: $mode"; exit 1 ;;
  esac
  echo "[wait] letting pipelines initialise (Spark ~15s) ..."; sleep 25

  # 1) throughput / latency / F1  (run sends its own warmup+load, REP repeats)
  echo "[run] throughput/latency, mode=$mode, repeats=$REP ..."
  ( cd "$MACJ" && python scripts/benchmark_distributed.py run \
      --mode "$mode" --duration "$DUR" --rate "$RATE" --warmup "$WARM" --repeats "$REP" \
      --output "$OUT/run_${mode}_${T}.json" )

  # 2) energy window (node-power on the active node(s))
  case "$mode" in
    single)        energy_window "$T" "$J2:jetson-nano-2:full" ; ACTIVE="$J2" ;;
    split|spark_cluster)
                   energy_window "$T" "$J1:jetson-nano-1:anomaly_gate" "$J2:jetson-nano-2:classifier"; ACTIVE="$J1 $J2" ;;
    horizontal)    energy_window "$T" "$J1:jetson-nano-1:full" "$J2:jetson-nano-2:full"; ACTIVE="$J1 $J2" ;;
  esac

  sync_logs "$T" $ACTIVE
  stop_all
  echo "[done] mode=$mode  ->  $OUT/run_${mode}_${T}.json (+ power_*_${T}.json)"
}

case "${1:-}" in
  single|split|horizontal|spark_cluster) run_mode "$1" ;;
  merge)
    ensure_mac_deps
    ( cd "$MACJ" && python scripts/benchmark_distributed.py merge \
        --input "$OUT/*.json" --output-csv "$OUT/summary.csv" )
    echo "[merge] -> $OUT/summary.csv" ;;
  *)
    echo "Usage: $0 {single|split|horizontal|spark_cluster|merge}"; exit 1 ;;
esac
