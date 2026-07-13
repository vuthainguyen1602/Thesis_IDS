#!/bin/bash
# Run edge benchmarks for SOICT paper (execute on Jetson or PC orchestrator)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$PAPER_DIR/results/benchmarks"
mkdir -p "$OUT"

cd "$ROOT/jetson"

if [ ! -d "venv" ]; then
    echo "[WARN] venv not found. Run: ./scripts/setup_jetson.sh"
fi

NODE_ID="${EDGE_NODE_ID:-$(hostname)}"
ROLE="${EDGE_NODE_ROLE:-full}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "================================================================"
echo "  SOICT 2026 — Edge benchmark"
echo "  Node: $NODE_ID  Role: $ROLE"
echo "================================================================"

MODE="${1:-local}"
shift || true

case "$MODE" in
  local)
    OUT_FILE="$OUT/local_${NODE_ID}_${STAMP}.json"
    python scripts/benchmark_distributed.py local \
        --mode "$ROLE" \
        --samples "${BENCHMARK_SAMPLES:-500}" \
        --output "$OUT_FILE" \
        "$@"
    ;;
  send)
    python scripts/benchmark_distributed.py send \
        --duration "${BENCHMARK_DURATION:-60}" \
        --rate "${BENCHMARK_RATE:-100}" \
        --output "$OUT/send_${STAMP}.json" \
        "$@"
    ;;
  collect)
    python scripts/benchmark_distributed.py collect \
        --mode "${BENCHMARK_DEPLOY_MODE:-split}" \
        --window-minutes "${BENCHMARK_WINDOW:-5}" \
        --output "$OUT/collect_${STAMP}.json" \
        "$@"
    ;;
  run)
    DEPLOY="${BENCHMARK_DEPLOY_MODE:-split}"
    OUT_FILE="$OUT/run_${DEPLOY}_${STAMP}.json"
    # Warmup is REAL traffic (excluded from the measured window); each config is
    # repeated (default 5) and merge reports mean±std. RAW_LATENCY_GLOB must
    # point at the RAW_LATENCY_LOG CSVs synced from the nodes for a publishable
    # run-level p95 (see jetson/JETSON_DISTRIBUTED.md).
    python scripts/benchmark_distributed.py run \
        --mode "$DEPLOY" \
        --duration "${BENCHMARK_DURATION:-60}" \
        --rate "${BENCHMARK_RATE:-100}" \
        --warmup "${BENCHMARK_WARMUP:-30}" \
        --repeats "${BENCHMARK_REPEATS:-5}" \
        ${RAW_LATENCY_GLOB:+--raw-latency-glob "$RAW_LATENCY_GLOB"} \
        --output "$OUT_FILE" \
        "$@"
    ;;
  node-power)
    # Run on EACH Jetson while the orchestrator drives the load window.
    python scripts/benchmark_distributed.py node-power \
        --duration "${BENCHMARK_POWER_DURATION:-90}" \
        --output "$OUT/power_${NODE_ID}_${STAMP}.json" \
        "$@"
    ;;
  merge)
    python scripts/benchmark_distributed.py merge \
        --input "$OUT/*.json" \
        --output-csv "$OUT/summary.csv"
    ;;
  *)
    echo "Usage: $0 {local|send|collect|run|node-power|merge} [extra args]"
    echo ""
    echo "Examples:"
    echo "  $0 local                          # micro-benchmark on this node"
    echo "  BENCHMARK_DEPLOY_MODE=split $0 run  # end-to-end (pipelines must be up)"
    echo "  $0 merge                          # merge all JSON -> summary.csv"
    exit 1
    ;;
esac

echo ""
echo "Distributed benchmark checklist:"
echo "  [ ] On EACH Jetson: export RAW_LATENCY_LOG=~/ids_raw_latency_\$(hostname).csv (before starting pipelines)"
echo "  [ ] On EACH Jetson (during runs): ./papers/soict2026/run_benchmarks.sh node-power"
echo "  [ ] Mode A (split): anomaly_gate on Jetson #1 + classifier on Jetson #2"
echo "  [ ] Mode B (horizontal): full on both nodes, same KAFKA_GROUP_ID"
echo "  [ ] Mode C (spark_cluster): start_spark_master/worker.sh"
echo "  [ ] Load sweep: BENCHMARK_RATE=50|100|200 (report saturation behaviour)"
echo "  [ ] Sync raw logs to orchestrator, set RAW_LATENCY_GLOB before 'run'/'collect'"
echo "  [ ] After all runs: ./papers/soict2026/run_benchmarks.sh merge (mean±std per mode)"
echo ""
