#!/usr/bin/env bash
# One pipeline-split run that replays the whole 10,000-flow test sample once
# (20 flows/s x 500 s, below the 22.4 flows/s sustained rate), so end-to-end
# precision / recall / F1 cover every replayed flow. Jetson #1's k3s is stopped
# for the run and started again afterwards, as in run_journal_sweep.sh.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SSH1="ssh -o ConnectTimeout=10 -o HostKeyAlias=192.168.1.50 bvdung@192.168.1.81"
log() { echo "[$(date +%H:%M:%S)] $*"; }
restore() {
  log "Jetson #1: starting k3s again"
  $SSH1 'sudo -n systemctl start k3s; sleep 15; echo "  k3s: $(systemctl is-active k3s)"' || true
  touch "$DIR/DONE"
}
trap restore EXIT
log "Jetson #1: stopping k3s for the run"
$SSH1 'sudo -n systemctl stop k3s; sudo -n /usr/local/bin/k3s-killall.sh >/dev/null 2>&1 || true; sleep 10; echo "  k3s: $(systemctl is-active k3s)"; uptime'
RATE=20 DUR=500 WARM=15 REP=1 COOL=30 NO_ENERGY=1 OUT_DIR="$DIR/runs" "$DIR/../../run_dist_bench.sh" split
