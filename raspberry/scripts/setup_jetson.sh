#!/bin/bash
# ==============================================================================
# Jetson Nano Setup Script - Distributed IDS Edge
# ==============================================================================
# Run on each Jetson Nano (Ubuntu 18.04/20.04 + JetPack).
# Usage: chmod +x setup_jetson.sh && ./setup_jetson.sh
# ==============================================================================

set -e

echo "================================================================"
echo "  IDS DISTRIBUTED EDGE - Jetson Nano Setup"
echo "================================================================"

echo "[1/7] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

echo "[2/7] Installing Python and build tools..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    git \
    htop \
    curl

echo "[3/7] Installing Java JDK (required for PySpark)..."
if ! command -v java &> /dev/null; then
    sudo apt-get install -y default-jdk
fi
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
grep -qxF "export JAVA_HOME=$JAVA_HOME" ~/.bashrc || echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
echo "  JAVA_HOME=$JAVA_HOME"

echo "[4/7] Configuring Jetson performance (optional but recommended)..."
if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks
    echo "  [OK] jetson_clocks enabled"
else
    echo "  [WARN] jetson_clocks not found (skip on non-Jetson systems)"
fi

echo "[5/7] Configuring swap (4GB recommended for PySpark)..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -qxF "/swapfile none swap sw 0 0" /etc/fstab || echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
    echo "  [OK] 4GB swap enabled"
else
    echo "  [OK] Swap already configured"
fi

echo "[6/7] Setting up Python virtual environment..."
cd "$(dirname "$0")/.."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ -f ../cluster/requirements_ml_driver_min.txt ]; then
    pip install --retries 10 --timeout 120 -r ../cluster/requirements_ml_driver_min.txt
fi
echo "  [OK] Python venv created"

echo "[7/7] Environment file..."
if [ ! -f .env ]; then
    cp .env.jetson1.example .env
    echo "  [OK] Created .env from .env.jetson1.example"
    echo "  Edit .env: set EDGE_NODE_ID, EDGE_NODE_ROLE, and Mac/PC IP addresses"
else
    echo "  [OK] .env already exists"
fi

echo ""
echo "================================================================"
echo "  SETUP COMPLETE"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Copy model artifacts to ~/raspberry/model/"
echo "  2. Configure .env for this node (jetson1 or jetson2 profile)"
echo "  3. Start pipeline: source venv/bin/activate && python edge/kafka_consumer.py"
echo ""
echo "Distributed modes:"
echo "  A) Horizontal scaling: both Jetsons use EDGE_NODE_ROLE=full, same KAFKA_GROUP_ID"
echo "  B) Pipeline split: jetson1=anomaly_gate, jetson2=classifier"
echo "  C) Spark cluster: jetson1 runs start_spark_master.sh, jetson2 runs start_spark_worker.sh"
echo ""
