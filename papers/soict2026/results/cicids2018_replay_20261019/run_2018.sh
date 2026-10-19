#!/usr/bin/env bash
# Pipeline split (gate on Jetson #1, PySpark classifier on Jetson #2), models
# trained on CICIDS2017, fed CSE-CIC-IDS2018 flows instead: a capacity sweep
# (2 repeats per rate, 45 s windows), then one run that replays all 10,000
# sampled 2018 flows once at the highest sustained rate. Jetson #1's k3s is
# stopped for the measurements and started again afterwards.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SOICT="$(cd "$DIR/../.." && pwd)"
JS="$(cd "$SOICT/../../jetson" && pwd)"
export REPLAY_CSV="$JS/sender/replay_cicids2018.csv"
export REPLAY_LABELS="$JS/sender/replay_cicids2018_labels.csv"
export OUT_DIR="$DIR/runs"
SSH1="ssh -o ConnectTimeout=10 -o HostKeyAlias=192.168.1.50 bvdung@192.168.1.81"
log() { echo "[$(date +%H:%M:%S)] $*"; }
restore() {
  log "Jetson #1: starting k3s again"
  $SSH1 'sudo -n systemctl start k3s; sleep 15; echo "  k3s: $(systemctl is-active k3s)"' || true
  touch "$DIR/DONE"
}
trap restore EXIT
log "Jetson #1: stopping k3s"
$SSH1 'sudo -n systemctl stop k3s; sudo -n /usr/local/bin/k3s-killall.sh >/dev/null 2>&1 || true; sleep 10; echo "  k3s: $(systemctl is-active k3s)"'

MODES=split RATES_split="${RATES:-5 8 12}" "$SOICT/run_capacity_sweep.sh"

# Highest rate whose every repeat drained within 10 s (else the lowest rate).
RATE=$(python - "$OUT_DIR/capacity_sweep.csv" <<'PY'
import csv, sys, collections
by = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    by[float(r["rate"])].append(float(r["drain_s"]))
ok = [k for k, v in by.items() if max(v) <= 10]
print(int(max(ok) if ok else min(by)))
PY
)
DUR=$(( (10000 + RATE - 1) / RATE ))
log "full replay: $RATE flows/s for $DUR s"
RATE=$RATE DUR=$DUR WARM=15 REP=1 COOL=30 NO_ENERGY=1 "$SOICT/run_dist_bench.sh" split
