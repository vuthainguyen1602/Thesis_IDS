#!/bin/bash
# Verify Spark master, workers, and synced parquet on cluster
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$CLUSTER_DIR/spark_cluster.env" ]; then
    # shellcheck disable=SC1091
    source "$CLUSTER_DIR/load_cluster_env.sh"
fi

MAC="${MAC_IP:-$(ipconfig getifaddr en0 2>/dev/null || echo unknown)}"
MASTER="${SPARK_MASTER:-spark://${MAC}:7077}"
WEBUI="${SPARK_MASTER_WEBUI:-http://${MAC}:8080}"

echo "================================================================"
echo "  Cluster health check"
echo "================================================================"
echo "  Master URL: $MASTER"
echo "  Master UI:  $WEBUI"
echo ""

if curl -sf "${WEBUI%/}/" >/dev/null 2>&1; then
    echo "[OK] Master UI reachable"
    curl -sf "${WEBUI}/json/" | python3 -c "
import json,sys
d=json.load(sys.stdin)
workers=d.get('workers',[])
alive=[w for w in workers if w.get('state')=='ALIVE']
print(f'  Workers ALIVE: {len(alive)}')
for w in alive:
    print(f\"    - {w.get('id','?')} @ {w.get('host','?')} ({w.get('cores',0)} cores)\")
" 2>/dev/null || echo "  (parse UI JSON skipped)"
else
    echo "[WARN] Master UI not reachable — run ./cluster/start_master_mac.sh"
fi

check_data() {
    local ssh_target="$1"
    local path="${IDS_CLUSTER_DATA_DIR:-/home/jetson/Thesis_IDS/data}"
    if ssh "$ssh_target" "test -f '${path}/train_data.parquet'"; then
        echo "[OK] Parquet on $ssh_target: $path"
    else
        echo "[ERR] Missing parquet on $ssh_target — run ./cluster/sync_workspace.sh"
    fi
}

for h in "${JETSON1_SSH:-}" "${JETSON2_SSH:-}"; do
    [ -n "$h" ] && check_data "$h"
done

echo ""
