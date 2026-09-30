#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Capacity sweep: for each deployment mode, raise the offered rate until the
# pipeline stops keeping up, and record queue-aware metrics per rate.
# RUN ON THE MAC, with the Docker services up and both Jetsons reachable.
#
#   ./run_capacity_sweep.sh                      # all modes below
#   MODES="split single_gate" ./run_capacity_sweep.sh
#
# A rate counts as sustained when every flow sent in the load window got its
# verdict within DRAIN_OK seconds of the window's end (default 10 s, about the
# time one PySpark batch takes on a Jetson). The sweep for a mode
# stops at the first rate that is not sustained. Results are appended to
# results/benchmarks/capacity_sweep.csv by analyze_runs.py.
# spark_cluster needs the Spark master and workers up and prompts for ENTER in
# run_dist_bench.sh, so it is not in the default list.
# ---------------------------------------------------------------------------
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/results/benchmarks"
CSV="$OUT/capacity_sweep.csv"
DRAIN_OK=${DRAIN_OK:-10}  # about one PySpark batch; flows done within it kept up
MODES=${MODES:-"single horizontal single_gate split"}

rates_for() {
  # RATES_<mode>="..." overrides the default list for one mode.
  local override="RATES_$1"
  if [ -n "${!override:-}" ]; then echo "${!override}"; return; fi
  case "$1" in
    single)       echo "1 2 3 4" ;;
    horizontal)   echo "2 4 6 8" ;;
    single_gate)  echo "5 10 20 30 40" ;;
    split)        echo "5 10 20 30 40" ;;
    spark_cluster) echo "5 10 20 30 40" ;;
  esac
}

for mode in $MODES; do
  for rate in $(rates_for "$mode"); do
    echo "================ $mode @ $rate flows/s ================"
    before=$(ls "$OUT"/run_"${mode}"_rep*.json 2>/dev/null | wc -l)
    NO_ENERGY=1 RATE=$rate REP=2 DUR=45 WARM=15 COOL=${COOL:-40} "$DIR/run_dist_bench.sh" "$mode"
    runs=$(ls -t "$OUT"/run_"${mode}"_rep*.json | head -2)
    python "$DIR/analyze_runs.py" --rate "$rate" --csv "$CSV" $runs | tee /tmp/sweep_last.txt
    worst=$(python - <<PY
import ast
rows=[ast.literal_eval(l) for l in open('/tmp/sweep_last.txt') if l.startswith('{') and "'mode'" in l]
print(max(r['drain_s'] for r in rows) if rows else 1e9)
PY
)
    echo "[sweep] $mode @ $rate: worst drain ${worst}s (sustained if <= ${DRAIN_OK}s)"
    if python -c "import sys; sys.exit(0 if float('$worst') <= $DRAIN_OK else 1)"; then
      continue
    fi
    echo "[sweep] $mode saturates at $rate flows/s; next mode"
    break
  done
done
echo "[sweep] results -> $CSV"
