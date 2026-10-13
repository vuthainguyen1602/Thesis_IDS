#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Journal extension of the SOICT edge table: the placements the conference
# version lacks (gate + classifier as two processes on ONE board; horizontal
# with the gate on) and every placement again with the classifier on ONNX
# Runtime, to ask whether the gate is still worth its recall cost once the
# classifier is no longer the bottleneck.
#
# RUN ON THE MAC, kept awake:
#   nohup caffeinate -dis ./run_journal_sweep.sh > results/journal_<stamp>/sweep.log 2>&1 &
#
# For each mode: capacity sweep (run_capacity_sweep.sh, REP repeats per rate,
# stops at the first rate that does not drain within DRAIN_OK s), then one run
# with the energy window at the highest sustained rate. Board clocks and ping
# RTT are recorded around every mode (clock_*.json) and taken off the latency.
# Jetson #1 hosts another project's k3s cluster; with QUIET_J1=1 (default) it
# is stopped for the measurements and started again at the end (also on
# Ctrl-C). Tunables: MODES, STAMP, OUT_DIR, DRAIN_OK, REP, QUIET_J1.
# ---------------------------------------------------------------------------
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
MACJ="$ROOT/jetson"
STAMP=${STAMP:-$(date +%Y%m%d)}
BASE="$DIR/results/journal_${STAMP}"
OUT="${OUT_DIR:-$BASE/runs}"; mkdir -p "$OUT"; export OUT_DIR="$OUT"
MODES=${MODES:-"split_1board horizontal_gate single_onnx single_gate_onnx split_1board_onnx split_onnx horizontal_onnx horizontal_gate_onnx"}
DRAIN_OK=${DRAIN_OK:-10}; export DRAIN_OK
REP=${REP:-2}; export REP
QUIET_J1=${QUIET_J1:-1}

J1=${J1:-192.168.1.81}; J2=${J2:-192.168.1.39}
J1_KEY=192.168.1.50; J2_KEY=192.168.1.204
U=bvdung
SSH1="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o HostKeyAlias=$J1_KEY $U@$J1"
SSH2="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o HostKeyAlias=$J2_KEY $U@$J2"
log() { echo "[$(date +%H:%M:%S)] $*"; }

j1_quiet() {
  log "Jetson #1: stopping k3s (+ its pods) and Docker containers for the measurements"
  $SSH1 'sudo -n systemctl stop k3s; sudo -n /usr/local/bin/k3s-killall.sh >/dev/null 2>&1 || true
         ids=$(sudo -n docker ps -q); [ -n "$ids" ] && sudo -n docker stop $ids >/dev/null
         pkill -f playwright 2>/dev/null; sleep 10
         echo "  k3s: $(systemctl is-active k3s)  docker containers: $(sudo -n docker ps -q | wc -l)"
         echo "  $(uptime)"; free -m | sed -n 2p'
}
j1_restore() {
  [ "$QUIET_J1" = "1" ] || return 0
  log "Jetson #1: starting k3s again"
  $SSH1 'sudo -n systemctl start k3s; sleep 15; echo "  k3s: $(systemctl is-active k3s)"; sudo -n k3s kubectl get pods -A --no-headers 2>/dev/null | awk "{print \$4}" | sort | uniq -c | sed "s/^/  pods /"' || true
}
board_state() {  # label
  for H in "$SSH1" "$SSH2"; do
    $H 'echo "$(hostname -I | cut -d" " -f1): $(sudo -n nvpmodel -q 2>/dev/null | tail -1) temp=$(cat /sys/devices/virtual/thermal/thermal_zone0/temp)mC load=$(cut -d" " -f1-3 /proc/loadavg) mem_used=$(free -m | awk "/Mem/{print \$3}")MB iwconfig=$(iwconfig 2>/dev/null | grep -o "Bit Rate=[0-9.]* Mb/s" | head -1)"'
  done
}

# Highest rate of a mode whose every repeat drained within DRAIN_OK s; else its lowest rate.
sustained_rate() {
  python - "$OUT/capacity_sweep.csv" "$1" "$DRAIN_OK" <<'PY'
import csv, sys
path, mode, ok = sys.argv[1], sys.argv[2], float(sys.argv[3])
worst = {}
for r in csv.DictReader(open(path)):
    if r["mode"] == mode:
        rate = float(r["rate"]); worst[rate] = max(worst.get(rate, 0.0), float(r["drain_s"]))
good = [r for r, d in worst.items() if d <= ok]
print(int(max(good)) if good else int(min(worst)) if worst else 0)
PY
}

trap 'log "interrupted"; "$DIR/run_dist_bench.sh" merge >/dev/null 2>&1; j1_restore; exit 130' INT TERM

log "journal sweep -> $OUT"
log "modes: $MODES  (REP=$REP, DRAIN_OK=${DRAIN_OK}s)"
{ echo "start $(date -u +%FT%TZ)"; sntp time.apple.com 2>&1 | tail -1; } > "$BASE/mac_clock_sntp.txt"
log "Mac clock vs NTP: $(tail -1 "$BASE/mac_clock_sntp.txt")"
[ "$QUIET_J1" = "1" ] && j1_quiet
board_state > "$BASE/boards_before.txt"; cat "$BASE/boards_before.txt"
( cd "$MACJ" && python scripts/clock_offset.py -n 40 --ping 30 --out "$BASE/clock_start.json" "$U@$J1:$J1_KEY" "$U@$J2:$J2_KEY" >/dev/null )
$SSH1 "ping -c 20 -i 0.2 $J2 | tail -2" > "$BASE/rtt_j1_to_j2.txt" 2>&1
log "RTT Jetson#1 -> Jetson#2: $(tail -1 "$BASE/rtt_j1_to_j2.txt")"

for mode in $MODES; do
  log "===== $mode: capacity sweep ====="
  MODES="$mode" "$DIR/run_capacity_sweep.sh"
  rate=$(sustained_rate "$mode")
  if [ "$rate" = "0" ]; then log "$mode: no rows in capacity_sweep.csv, skipping the energy window"; continue; fi
  log "===== $mode: energy window at $rate flows/s ====="
  RATE=$rate REP=1 DUR=45 WARM=15 COOL=30 "$DIR/run_dist_bench.sh" "$mode"
  run=$(ls -t "$OUT"/run_"${mode}"_*.json | grep -v _rep | head -1)
  clocks=$(ls -t "$OUT"/clock_"${mode}"_*.json | head -2)
  python "$DIR/analyze_runs.py" "$run" --rate "$rate" --csv "$OUT/energy_runs.csv" --clock-json $clocks
  board_state
done

( cd "$MACJ" && python scripts/clock_offset.py -n 40 --ping 30 --out "$BASE/clock_end.json" "$U@$J1:$J1_KEY" "$U@$J2:$J2_KEY" >/dev/null )
{ echo "end $(date -u +%FT%TZ)"; sntp time.apple.com 2>&1 | tail -1; } >> "$BASE/mac_clock_sntp.txt"
board_state > "$BASE/boards_after.txt"
"$DIR/run_dist_bench.sh" merge
j1_restore
trap - INT TERM
log "DONE -> $OUT/capacity_sweep.csv, $OUT/energy_runs.csv"
touch "$BASE/DONE"
