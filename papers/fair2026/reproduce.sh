#!/bin/bash
# FAIR'2026 — always distributed (Mac Spark master + 2× Jetson workers)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/cluster/reproduce_cluster.sh" fair
