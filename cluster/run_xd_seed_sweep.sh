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
#
# Per-run results land in results/ml_11_cross_dataset/sweep/<label>.csv on the
# Mac; the summary at the end is what answers the question.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/load_cluster_env.sh"

OUT="$ROOT/results/ml_11_cross_dataset/sweep"
LOGS="$ROOT/output/xd_sweep"
mkdir -p "$OUT" "$LOGS"

SEEDS="${SEEDS:-42 42 7 13}"
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
    for host in "${JETSON1_SSH:?}" ${JETSON2_ENABLED:+${JETSON2_SSH:-}}; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
            'cd ~/Thesis_IDS && unset SPARK_HOME && setsid nohup ./cluster/start_worker.sh \
             > /tmp/start_worker.log 2>&1 < /dev/null & disown' 2>/dev/null || true
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
    label="run${i}_seed${seed}"
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
echo "  Summary — cross-dataset F1 per run"
echo "================================================================"
python3 - "$OUT" <<'PY'
import csv, glob, os, statistics, sys
rows = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.csv"))):
    label = os.path.basename(path)[:-4]
    for r in csv.DictReader(open(path)):
        if r["kind"] == "cross":
            rows.setdefault(f'{r["train"]}->{r["test"]}', []).append((label, float(r["f1"])))
for pair, vals in rows.items():
    print(f"\n{pair}")
    for label, f1 in vals:
        print(f"   {label:16s} {f1:.6f}")
    fs = [f for _, f in vals]
    if len(fs) > 1:
        print(f"   {'mean':16s} {statistics.mean(fs):.6f}"
              f"   sd {statistics.pstdev(fs):.6f}   range {max(fs) - min(fs):.6f}")
same = [f for l, f in rows.get("CICIDS2017->CSE-CIC-IDS2018", []) if l.endswith("seed42")]
if len(same) >= 2:
    verdict = ("NOT deterministic — identical configuration gave different results"
               if max(same) - min(same) > 1e-9 else
               "deterministic under a fixed seed — differences across seeds are seed sensitivity")
    print(f"\nVerdict on the repeated seed-42 runs: {verdict}")
PY
