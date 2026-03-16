#!/bin/bash
# ==============================================================================
# Raspberry Pi Setup Script - IDS Edge Deployment
# ==============================================================================
# Run this script on a fresh Raspberry Pi OS (64-bit) to set up the environment.
# Usage: chmod +x setup_raspberry.sh && ./setup_raspberry.sh
# ==============================================================================

set -e

echo "================================================================"
echo "  IDS EDGE DEPLOYMENT - Raspberry Pi Setup"
echo "  Mode: Pipeline only (Kafka + DB on Mac/PC)"
echo "================================================================"

# --- System Update ---
echo "[1/6] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# --- Python 3.9+ ---
echo "[2/6] Installing Python and build tools..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    git

# --- Docker (optional, only needed if running Kafka/DB on RPi) ---
echo "[3/6] Docker..."
read -p "  Install Docker on RPi? (y/N, skip if Docker runs on Mac): " INSTALL_DOCKER
if [[ "$INSTALL_DOCKER" =~ ^[Yy]$ ]]; then
    if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        echo "  [OK] Docker installed"
    else
        echo "  [OK] Docker already installed"
    fi
else
    echo "  [OK] Skipped (Docker runs on Mac/PC)"
fi

# --- Java JDK (required for PySpark) ---
echo "[4/6] Installing Java JDK..."
if ! command -v java &> /dev/null; then
    sudo apt-get install -y default-jdk
    echo "  [OK] Java JDK installed"
else
    java_version=$(java -version 2>&1 | head -n1)
    echo "  [OK] Java already installed: $java_version"
fi

# Set JAVA_HOME
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
echo "  JAVA_HOME=$JAVA_HOME"

# --- Python Virtual Environment ---
echo "[5/6] Setting up Python virtual environment..."
cd "$(dirname "$0")/.."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "  [OK] Python venv created and dependencies installed"

# --- Environment File ---
echo "[6/6] Creating .env from template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  [OK] Created .env file - please update with your settings"
else
    echo "  [OK] .env file already exists"
fi

echo ""
echo "================================================================"
echo "  SETUP COMPLETE"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Update .env with the Mac's IP address:"
echo "     nano .env"
echo "     KAFKA_BOOTSTRAP_SERVERS=<mac-ip>:9092"
echo "     POSTGRES_HOST=<mac-ip>"
echo "     INFLUXDB_URL=http://<mac-ip>:8086"
echo ""
echo "  2. Copy the trained model from the training machine:"
echo "     scp user@<mac-ip>:~/Thesis_IDS/raspberry/model/* model/"
echo ""
echo "  3. Start the IDS pipeline:"
echo "     source venv/bin/activate"
echo "     python edge/kafka_consumer.py"
echo ""
echo "  4. On Mac: docker compose up -d && python sender/data_sender.py"
echo ""
