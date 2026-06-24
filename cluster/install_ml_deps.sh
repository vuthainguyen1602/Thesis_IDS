#!/bin/bash
# Install ML Python deps on the current host (Spark driver AND executors need this).
# Run locally on a Jetson, or via: ssh user@jetson 'IDS_ROOT=~/Thesis_IDS bash -s' < cluster/install_ml_deps.sh
set -euo pipefail

if [ -n "${IDS_ROOT:-}" ] && [ -d "$IDS_ROOT" ]; then
    ROOT="$IDS_ROOT"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
    if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/install_ml_deps.sh" ]; then
        ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    else
        ROOT="${HOME}/Thesis_IDS"
    fi
fi

cd "$ROOT"
if [ -d jetson/venv/bin ]; then
    source jetson/venv/bin/activate
elif [ -d venv/bin ]; then
    source venv/bin/activate
else
    echo "[ERR] No venv found under $ROOT (run jetson/scripts/setup_jetson.sh first)" >&2
    exit 1
fi

PIP_OPTS="--retries 10 --timeout 120 --default-timeout=120"
REQ_FULL="cluster/requirements_ml_driver.txt"

_needs_ml_deps() {
    python -c "import pandas, matplotlib, seaborn, pyarrow, xgboost, shap" 2>/dev/null || return 0
    return 1
}

if _needs_ml_deps; then
    echo "[INFO] Installing ML deps (pyarrow, xgboost, shap, ...) on $(hostname)..."
    if [ -f "$REQ_FULL" ]; then
        pip install $PIP_OPTS -r "$REQ_FULL" || {
            echo "[WARN] Batch install failed — trying one package at a time..."
            pip install $PIP_OPTS pandas matplotlib seaborn pyarrow
            pip install $PIP_OPTS xgboost shap
        }
    else
        pip install $PIP_OPTS pandas matplotlib seaborn pyarrow xgboost shap
    fi
    pip install $PIP_OPTS synapseml 2>/dev/null || true
else
    echo "[OK] ML deps already installed on $(hostname)"
fi

python -c "import pyarrow, xgboost; print('[OK] pyarrow + xgboost on', __import__('sys').executable)"
