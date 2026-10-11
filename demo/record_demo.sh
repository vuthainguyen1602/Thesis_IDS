#!/usr/bin/env bash
# Record the thesis demo video: the split edge pipeline running live, seen
# through the Grafana dashboard. Run on the Mac with the Docker stack up.
#
#   ./demo/record_demo.sh                 # writes demo/out/<tag>/ and output/videos/
#   RATE=15 SEND_SECONDS=150 ./demo/record_demo.sh
#
# Steps: start the anomaly gate on Jetson #1 and the PySpark classifier on
# Jetson #2 (fresh Kafka consumer groups), start recording the dashboard,
# replay CICIDS2017 flows at RATE flows/s for SEND_SECONDS, keep recording while
# the classifier drains, stop both nodes, then build the video with captions
# timed from the logged events (demo/make_video.sh).
set -euo pipefail

J1=${J1:-192.168.1.81}; J2=${J2:-192.168.1.39}; U=bvdung
RATE=${RATE:-15}; SEND_SECONDS=${SEND_SECONDS:-150}
LEAD=${LEAD:-20}; TAIL=${TAIL:-40}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAC_IP=${MAC_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)}
TAG=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/demo/out/$TAG"
mkdir -p "$OUT"
JROOT=/home/$U/Thesis_IDS/jetson
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
REMOTE_ENV="KAFKA_BOOTSTRAP_SERVERS=$MAC_IP:9092 POSTGRES_HOST=$MAC_IP POSTGRES_PORT=5433 \
INFLUXDB_URL=http://$MAC_IP:8086 KAFKA_GROUP_ID=ids-demo-$TAG KAFKA_CLASSIFIER_GROUP_ID=ids-demo-clf-$TAG"

stop_nodes() {
  # [k] keeps pkill from matching the ssh shell that carries the pattern.
  $SSH "$U@$J1" "pkill -f '[k]afka_consumer.py' || true"
  $SSH "$U@$J2" "pkill -f '[k]afka_consumer.py' || true"
}

start_node() {  # host node_id role alert
  # Background the ssh client itself (as run_dist_bench.sh does): the remote
  # nohup'd job can keep the channel open and would otherwise block here.
  $SSH -n "$U@$1" "cd $JROOT && source venv/bin/activate && \
    $REMOTE_ENV EDGE_NODE_ID=$2 EDGE_NODE_ROLE=$3 ALERT_ENABLED=$4 SPARK_MASTER= \
    nohup python edge/kafka_consumer.py > \$HOME/edge_demo_$2.log 2>&1 < /dev/null &" &
  sleep 5
}

# macOS date has no %N, so take the epoch from Python.
event() { echo "$(python3 -c 'import time; print(f"{time.time():.3f}")') $1" >> "$OUT/events.txt"; echo "[demo] $1"; }

REC=""
cleanup() { [ -n "$REC" ] && kill "$REC" 2>/dev/null || true; stop_nodes; }
trap cleanup EXIT

stop_nodes; sleep 3
start_node "$J1" jetson-nano-1 anomaly_gate 0
start_node "$J2" jetson-nano-2 classifier 1
echo "[demo] waiting for the PySpark classifier to load ..."
for _ in $(seq 1 90); do
  $SSH "$U@$J2" "grep -q -i -E 'Spark Session created|PipelineModel|listening|consum' \$HOME/edge_demo_jetson-nano-2.log" && break
  sleep 2
done
sleep 10

python "$ROOT/demo/record_grafana.py" --seconds $((LEAD + SEND_SECONDS + TAIL)) --out-dir "$OUT" > "$OUT/record.log" 2>&1 &
REC=$!
until [ -f "$OUT/frames/00000.png" ]; do sleep 0.5; done
event "record_start"
sleep "$LEAD"
event "send_start"
# Mac-side env as in run_dist_bench.sh: config.py reads these before jetson/.env.
CSV="$ROOT/jetson/sender/replay_cicids2017.csv"
( cd "$ROOT/jetson" && KAFKA_BOOTSTRAP_SERVERS=localhost:9092 POSTGRES_HOST=localhost POSTGRES_PORT=5433 \
    INFLUXDB_URL=http://localhost:8086 DATA_CSV_PATH="$CSV" python scripts/benchmark_distributed.py send \
    --csv "$CSV" --duration "$SEND_SECONDS" --rate "$RATE" --output "$OUT/send.json" > "$OUT/send.log" 2>&1 )
event "send_end"
wait $REC; REC=""
event "record_end"
scp -q "$U@$J1:edge_demo_jetson-nano-1.log" "$U@$J2:edge_demo_jetson-nano-2.log" "$OUT/" || true

"$ROOT/demo/make_video.sh" "$OUT"
