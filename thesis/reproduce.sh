#!/bin/bash
# Thesis — always distributed (Mac + 2× Jetson)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/cluster/reproduce_cluster.sh" thesis
