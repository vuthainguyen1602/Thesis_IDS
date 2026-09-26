#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cross-dataset stability sweep.
#
# The cross-dataset F1 for CICIDS2017 -> CSE-CIC-IDS2018 has landed on two very
# different values across runs of the same code on the same parquet: 0.042270
# twice (bit-identical, and matching the published table) and 0.29-0.30 three
# times. No configuration difference found so far explains the split.
#
# This sweep is built to tell the two candidate explanations apart:
#
#   runs 1 and 2   same seed (42)      -> if they disagree, the pipeline is not
#                                         deterministic; partitioning decides
#   runs 3 and 4   seeds 7 and 13      -> if 1 == 2 but 3, 4 differ, the result
#                                         is seed-sensitive, not noisy
#
# Only the two base fits run: the adaptation and label-budget blocks are off,
# since the question is about the baseline number alone.
#
#   ./cluster/run_xd_seed_sweep.sh            # 4 runs, ~3-4h at 8 cores
#   SEEDS="42 42 7 13" ./cluster/run_xd_seed_sweep.sh
#   SEEDS="5 11 23 31" ./cluster/run_xd_seed_sweep.sh   # add 4 more draws
#
# Per-run results land in results/ml_11_cross_dataset/sweep/<label>.csv on the
# Mac; the summary at the end is what answers the question. Repetitions
# accumulate in that directory, so a later night can add runs and re-summarise
# everything with  python3 cluster/xd_sweep_summary.py  on its own.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/load_cluster_env.sh"

OUT="$ROOT/results/ml_11_cross_dataset/sweep"
LOGS="$ROOT/output/xd_sweep"
mkdir -p "$OUT" "$LOGS"

SEEDS="${SEEDS:-42 42 7 13}"
# Repetitions from different nights have to accumulate, not overwrite each
# other, so each sweep writes under its own stamp.
BATCH="${BATCH:-$(date +%Y%m%d-%H%M)}"
REMOTE_CSV="${CLUSTER_DRIVER_IDS_ROOT}/results/ml_11_cross_dataset/cross_dataset_results.csv"

# The worker JVMs on the 8GB boards have not survived every heavy run — they
# were gone after three of the last four. An unattended sweep that does not
# notice loses the rest of the night, so bring them back between runs.
ensure_workers() {
    local want="${1:-2}" ui="${SPARK_MASTER_WEBUI:-http://${MAC_IP:?}:8080}" n
    n=$(curl -s -m 5 "${ui%/}/json/" 2>/dev/null \
        | python3 -c 'import sys,json;print(json.load(sys.stdin).get("aliveworkers",0))' 2>/dev/null || echo 0)
    [ "${n:-0}" -ge "$want" ] && { echo "[OK] ${n} worker(s) ALIVE"; return 0; }

    echo "[WARN] only ${n:-0} worker(s) ALIVE — restarting"
    # The ssh session itself is backgrounded: start_worker.sh has a habit of
    # holding the channel open even with nohup, and a blocked ssh would stall
    # the sweep indefinitely. The daemon is detached remotely with setsid, so
    # the master's own worker list is the thing to poll, not ssh's exit code.
    for host in "${JETSON1_SSH:?}" ${JETSON2_ENABLED:+${JETSON2_SSH:-}}; do
        ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host" \
            'cd ~/Thesis_IDS && unset SPARK_HOME && setsid nohup ./cluster/start_worker.sh \
             > /tmp/start_worker.log 2>&1 < /dev/null & disown' >/dev/null 2>&1 &
    done
    for _ in $(seq 1 12); do
        sleep 5
        n=$(curl -s -m 5 "${ui%/}/json/" 2>/dev/null \
            | python3 -c 'import sys,json;print(json.load(sys.stdin).get("aliveworkers",0))' 2>/dev/null || echo 0)
        [ "${n:-0}" -ge "$want" ] && { echo "[OK] ${n} worker(s) ALIVE"; return 0; }
    done
    echo "[ERR] workers did not come back (${n:-0} alive)"; return 1
}

i=0
for seed in $SEEDS; do
    i=$((i + 1))
    label="${BATCH}_run${i}_seed${seed}"
    echo ""
    echo "================================================================"
    echo "  Sweep $i/$(echo $SEEDS | wc -w | tr -d ' ')  —  seed=$seed  —  $label"
    echo "================================================================"

    ensure_workers 2 || { echo "[SKIP] $label — no workers"; continue; }

    IDS_XD_SEED="$seed" IDS_XD_ADAPT=0 IDS_XD_TARGET_LABEL_FRAC="" \
        "$HERE/run_ml_remote.sh" ml_11_cross_dataset_eval.py \
        > "$LOGS/$label.log" 2>&1 || {
            echo "[WARN] $label failed; see $LOGS/$label.log"; continue; }

    scp -q -o StrictHostKeyChecking=accept-new \
        "${CLUSTER_DRIVER}:${REMOTE_CSV}" "$OUT/$label.csv"
    echo "[OK] $label -> $OUT/$label.csv"
    grep -E "CSE-CIC-IDS2018,cross|CICIDS2017,cross" "$OUT/$label.csv" | cut -d, -f1-4
done

echo ""
echo "================================================================"
echo "  Summary — all runs in $OUT"
echo "================================================================"
# Kept in its own script so repetitions added on a later night are summarised
# by re-running it over the same directory, no sweep re-run needed.
python3 "$HERE/xd_sweep_summary.py" "$OUT"
