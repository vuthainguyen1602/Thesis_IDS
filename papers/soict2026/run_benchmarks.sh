#!/bin/bash
# Run edge benchmarks for SOICT paper (execute on Jetson or PC orchestrator)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$PAPER_DIR/results/benchmarks"
mkdir -p "$OUT"

cd "$ROOT/raspberry"

if [ ! -d "venv" ]; then
    echo "[WARN] venv not found. Run: ./scripts/setup_jetson.sh"
fi

NODE_ID="${EDGE_NODE_ID:-$(hostname)}"
ROLE="${EDGE_NODE_ROLE:-full}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
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
    python scripts/benchmark_distributed.py run \
        --mode "$DEPLOY" \
        --duration "${BENCHMARK_DURATION:-60}" \
        --rate "${BENCHMARK_RATE:-100}" \
        --warmup "${BENCHMARK_WARMUP:-15}" \
        --output "$OUT_FILE" \
        "$@"
    ;;
  merge)
    python scripts/benchmark_distributed.py merge \
        --input "$OUT/*.json" \
        --output-csv "$OUT/summary.csv"
    ;;
  *)
    echo "Usage: $0 {local|send|collect|run|merge} [extra args]"
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
echo "  [ ] Mode A (split): anomaly_gate on Jetson #1 + classifier on Jetson #2"
echo "  [ ] Mode B (horizontal): full on both nodes, same KAFKA_GROUP_ID"
echo "  [ ] Mode C (spark_cluster): start_spark_master/worker.sh"
echo "  [ ] PC: python sender/data_sender.py --rate 50|100|200"
echo "  [ ] After all runs: ./papers/soict2026/run_benchmarks.sh merge"
echo ""
