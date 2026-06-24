# IDS Edge Deployment — Detailed Guide

This guide covers deploying the **PySpark-based IDS** onto an **NVIDIA Jetson Orin Nano Super (8GB)** edge node using a **Split Deployment** architecture: Docker infrastructure (Kafka/Postgres/InfluxDB/Grafana) runs on the **Mac**, inference runs on the **edge**. A **Raspberry Pi 4B (8GB)** also works as an alternative single-node target — the edge code is identical; only the OS setup script differs.

> **Distributed 2× Jetson modes** (horizontal split, anomaly-gate + classifier roles) are documented separately in **[JETSON_DISTRIBUTED.md](JETSON_DISTRIBUTED.md)**. This guide covers a **single** edge node.

---

## ⚡ Quick Start Checklist

If you are already familiar with the setup, use these commands in order:

1. **Mac**: `cd raspberry && docker compose up -d` (Start Infra)
2. **Mac**: `python scripts/save_model.py` (Export trained model)
3. **Jetson**: `./scripts/setup_jetson.sh` (First time only; on RPi use `setup_raspberry.sh`)
4. **Jetson**: `scp -r mac_user@mac_ip:~/path/to/model ~/Thesis_IDS/raspberry/model/`
5. **Jetson**: `python edge/kafka_consumer.py` (Start Detection)
6. **Mac**: `python sender/data_sender.py` (Send stream)

---

## 🏗️ Split Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MAC / PC                              │
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │  Data Sender      │  │  Docker Compose              │ │
│  │  (CSV → Kafka)    │  │  ├── Kafka + Zookeeper       │ │
│  └──────────────────┘  │  ├── PostgreSQL               │ │
│                         │  ├── InfluxDB                 │ │
│                         │  └── Grafana (:3000)          │ │
│                         └──────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────┘
                      │  WiFi / Ethernet (same LAN)
                      │
┌─────────────────────┴───────────────────────────────────┐
│           JETSON ORIN NANO SUPER 8GB (or RPi 4B)         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Kafka Consumer → (Optional) Anomaly Gate (AE)    │   │
│  │  → Preprocessor → PySpark Model → Monitor → Alert │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## PART 1: SETUP ON MAC/PC

### Step 1.1: Install Docker Desktop

```bash
# Check if Docker is installed
docker --version

# If not installed, download from https://www.docker.com/products/docker-desktop/
# After installation, open Docker Desktop and wait for the green icon (running)
```

### Step 1.2: Start the Infrastructure

```bash
# Navigate to the raspberry directory
cd /Users/thainguyenvu/Desktop/Thesis_IDS/raspberry

# Start all services
docker compose up -d

# Wait ~30 seconds then verify
docker compose ps
```

**Expected output:**
```
NAME             STATUS    PORTS
ids-grafana      running   0.0.0.0:3000->3000/tcp
ids-influxdb     running   0.0.0.0:8086->8086/tcp
ids-kafka        running   0.0.0.0:9092->9092/tcp
ids-postgres     running   0.0.0.0:5432->5432/tcp
ids-zookeeper    running   0.0.0.0:2181->2181/tcp
```

### Step 1.3: Save the PySpark Model

```bash
# Still on Mac, run the model save script
cd /Users/thainguyenvu/Desktop/Thesis_IDS/raspberry
python scripts/save_model.py

# Result: the model/ directory will contain:
#   model/ids_pipeline_model/    (PySpark PipelineModel)
#   model/feature_columns.json   (list of 30 features)
```

### (Optional) Step 1.3b: Train & Export the Anomaly Gate (Autoencoder)

This project supports a **two-stage edge IDS**:

- **Stage A (Anomaly Gate)**: lightweight sklearn autoencoder (filters suspicious flows)
- **Stage B (Classifier)**: existing PySpark PipelineModel (final Attack/Benign decision)

Train/export the anomaly gate on your Mac/PC (project root):

```bash
cd /Users/thainguyenvu/Desktop/Thesis_IDS

# Requires feature list to be exported first (Step 1.3)
python ml_08_anomaly_gate_autoencoder.py

# Result: these new files will be created under raspberry/model/
#   raspberry/model/anomaly_autoencoder.pkl
#   raspberry/model/anomaly_scaler.pkl
#   raspberry/model/anomaly_threshold.json
```

### Step 1.4: Find the Mac's IP Address

```bash
# Get the Mac's LAN IP address
ifconfig | grep "inet " | grep -v 127.0.0.1if

# Example output: inet 192.168.1.100 netmask 0xffffff00
# Note this IP (e.g. 192.168.1.100)
```

---

## PART 2: SETUP ON THE JETSON (EDGE NODE)

### Step 2.1: Flash JetPack (Jetson Orin Nano Super)

```bash
# Flash JetPack 6.x (Ubuntu, aarch64) with NVIDIA SDK Manager or the SD-card image.
# Download: https://developer.nvidia.com/embedded/jetpack
# Boot the Jetson, complete first-boot setup, and connect it to the SAME LAN as the Mac.

# (Alternative target — Raspberry Pi 4B: flash Raspberry Pi OS 64-bit to a 32GB SD card.)
```

### Step 2.2: SSH into the Jetson

```bash
# From Mac, SSH into the Jetson
ssh <user>@<jetson-ip>
# (On RPi: ssh pi@<rpi-ip> — change the default password immediately.)
```

### Step 2.3: Clone the Project to the Jetson

```bash
# On the Jetson
cd ~
git clone <repo-url> Thesis_IDS
cd Thesis_IDS/raspberry/

# OR: Copy directly from Mac
# On Mac, run:
# scp -r /Users/thainguyenvu/Desktop/Thesis_IDS <user>@<jetson-ip>:~/Thesis_IDS
```

### Step 2.4: Run the Automated Setup Script

```bash
# On the Jetson
cd ~/Thesis_IDS/raspberry
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
# (On RPi: ./scripts/setup_raspberry.sh instead.)

# setup_jetson.sh will automatically:
# [1/7] Update system packages
# [2/7] Install Python3 + pip + venv + build tools
# [3/7] Install Java JDK (default-jdk, for PySpark)
# [4/7] Configure Jetson performance (jetson_clocks / power mode — optional)
# [5/7] Configure a 4GB safety swap file
# [6/7] Create Python venv + install deps (pyspark, kafka-python, scikit-learn, ...)
# [7/7] Create .env from the template

# Estimated time: ~10–15 minutes (Java + PySpark downloads take a while)
```

### Step 2.5: Configure .env

```bash
# On the Jetson, create .env from the Jetson template, then edit it
cp .env.jetson1.example .env    # node #2 uses .env.jetson2.example
nano .env

# Replace the default IP (192.168.1.100) with the Mac's actual IP:
KAFKA_BOOTSTRAP_SERVERS=<mac-ip>:9092     # e.g. 192.168.1.100:9092
POSTGRES_HOST=<mac-ip>                     # e.g. 192.168.1.100
INFLUXDB_URL=http://<mac-ip>:8086         # e.g. http://192.168.1.100:8086

# For Telegram alerts:
TELEGRAM_BOT_TOKEN=<bot-token>
TELEGRAM_CHAT_ID=<chat-id>

# Save: Ctrl+O → Enter → Ctrl+X
```

### Step 2.6: Copy the Model from Mac

```bash
# On the Jetson
scp -r <mac-user>@<mac-ip>:/Users/thainguyenvu/Desktop/Thesis_IDS/raspberry/model/ ~/Thesis_IDS/raspberry/model/

# Verify
ls -la ~/Thesis_IDS/raspberry/model/
# Should contain:
#   ids_pipeline_model/    (PySpark model directory)
#   feature_columns.json   (JSON file)
#
# If you enabled the anomaly gate, it should also contain:
#   anomaly_autoencoder.pkl
#   anomaly_scaler.pkl
#   anomaly_threshold.json
```

### Step 2.7: Test Connectivity

```bash
# On the edge node, verify the Mac is reachable
ping <mac-ip> -c 3

# Test Kafka
nc -zv <mac-ip> 9092     # Expected: Connection to ... succeeded!

# Test PostgreSQL
nc -zv <mac-ip> 5432     # Expected: Connection to ... succeeded!
```

---

## PART 3: RUNNING THE SYSTEM

### Step 3.1: Start IDS on the edge node

```bash
# On the Jetson — Terminal 1
cd ~/Thesis_IDS/raspberry
source venv/bin/activate
python edge/kafka_consumer.py

# Expected output:
# ============================================================
#   INITIALIZING IDS EDGE PIPELINE (PySpark)
# ============================================================
# [INFO] Spark Session created (version: 3.4.1)
# [INFO] Feature columns loaded: 30 features
# [INFO] PySpark Model loaded from ./model/ids_pipeline_model
# [INFO] PostgreSQL connected: 192.168.1.100:5432/ids_edge
# [INFO] InfluxDB connected: http://192.168.1.100:8086
# [INFO] Performance Monitor started (interval: 10s)
# [INFO] Kafka Consumer subscribed to 'ids-network-flow'
#
# ============================================================
#   IDS EDGE PIPELINE READY (PySpark) - Waiting for messages...
# ============================================================

# Note: Spark Session init takes ~10–15 s on Jetson Orin Nano Super (~20–30 s on RPi)
```

#### (Optional) Enable the anomaly gate on the edge node

By default, the anomaly gate is disabled (pipeline runs exactly as before).

To enable:

```bash
export ANOMALY_ENABLED=1
python edge/kafka_consumer.py
```

When enabled, you should see an additional line at startup:

- `[OK] AnomalyScorer loaded ... Threshold: ...`

### Step 3.2: Send Data from Mac

```bash
# On Mac - new Terminal window
cd /Users/thainguyenvu/Desktop/Thesis_IDS/raspberry

# Install kafka-python if not already installed
pip install kafka-python

# Stream test data (100 rows/second)
python sender/data_sender.py \
    --csv /Users/thainguyenvu/Desktop/roedunet-simargl2021 \
    --rate 100

# Output on Mac:
#   [INFO] Kafka Producer connected to localhost:9092
#   Sent 1,000 rows | Rate: 100.0 rows/s | Elapsed: 10.0s
#   Sent 2,000 rows | Rate: 100.1 rows/s | Elapsed: 20.0s
#   ...
```

### Step 3.3: Monitor Results on the edge node

```
# The edge node terminal will display:
  [100] Batch: 45ms | Attacks: 23/100 | Avg: 4.5ms
  [200] Batch: 42ms | Attacks: 51/200 | Avg: 4.2ms
  [300] Batch: 40ms | Attacks: 78/300 | Avg: 4.0ms

  [MONITOR] CPU: 35% | MEM: 42% (1680MB) | Throughput: 22.5 rps | Latency: 4.3ms | Attacks: 25
            Temp: 52.0°C
```

### Step 3.4: View the Grafana Dashboard

```bash
# Open browser on Mac
open http://localhost:3000

# Login: admin / admin
# Go to: Dashboards → Import → Upload JSON
# Select: raspberry/dashboard/grafana_dashboard.json

# The dashboard displays:
# - Throughput (req/s) in real time
# - Number of detected attacks
# - Prediction latency (ms)
# - Edge CPU / Memory usage
# - Edge CPU/GPU temperature
# - Recent alerts table
#
# If you imported the updated dashboard JSON, it also includes Postgres panels:
# - Prediction Routes (anomaly_gate_only vs spark_classifier)
# - Attack vs Benign counts from Postgres `predictions`
# - Recent Predictions table (route + anomaly score/threshold)
```

---

## PART 4: ALERTING CONFIGURATION (OPTIONAL)

The IDS supports real-time notifications via Email (Mailtrap) and Slack. These are triggered automatically when an attack is detected.

### 4.1: Setup Mailtrap (Email)
Mailtrap allows you to test SMTP email delivery without sending real emails to your personal inbox.

1.  **Register**: Create a free account at [mailtrap.io](https://mailtrap.io).
2.  **Get Credentials**: Go to **Inboxes** → **My Inbox** → **SMTP Settings**.
3.  **Configure `.env`**: Copy the `Username` and `Password` to your `.env` file on the edge node:
    ```env
    SMTP_USER=your_username
    SMTP_PASSWORD=your_password
    SMTP_HOST=sandbox.smtp.mailtrap.io
    SMTP_PORT=2525
    ALERT_EMAIL_TO=your-real-email@example.com
    ```

### 4.2: Setup Slack Webhook
Slack Webhooks allow the IDS to post messages directly to a Slack channel.

1.  **Create App**: Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**.
2.  **Enable Webhooks**: Navigate to **Incoming Webhooks** and toggle it to **On**.
3.  **Create Webhook**: Click **Add New Webhook to Workspace**, select a channel, and click **Allow**.
4.  **Configure `.env`**: Copy the **Webhook URL** to your `.env` file:
    ```env
    WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../X...
    ```

---

## PART 5: STOPPING THE SYSTEM

```bash
# On the edge node: Ctrl+C to stop the pipeline
# The pipeline will display final statistics:
#   Final Statistics:
#     Total predictions:  5,000
#     Total attacks:      1,250
#     Attack rate:        25.00%
#     Avg latency:        4.200 ms
#   [INFO] Pipeline shutdown complete.

# On Mac: Ctrl+C to stop the sender, then:
docker compose down     # Stop all services
```

---

## PART 6: ADVANCED OPTIONS

### 6.1: Benchmarking Multiple Models
If you want to compare different algorithms (Decision Tree, Random Forest, GBT) on the edge, use the "Save All" script:

```bash
# On Mac
python scripts/save_all_models.py

# This saves 3 different models to model/
# Update edge/prediction_engine.py or .env to point to the desired model path
```

### 6.2: Monitoring Edge Performance
On the Jetson, use `tegrastats` (CPU/GPU/RAM/power) or `htop` via SSH:

```bash
# On the Jetson
sudo tegrastats          # CPU/GPU load, RAM, power draw, temperatures
htop                     # per-core CPU + memory
```

For energy-per-inference measurements, `edge/power_monitor.py` samples
`tegrastats` while the pipeline runs (used by the SOICT edge benchmarks).

Watch for:
- **CPU/GPU Usage**: PySpark uses multiple cores during batch inference.
- **Memory**: Keep the resident set size (RSS) within physical RAM (leave ~1GB for the OS); `setup_jetson.sh` adds a 4GB swap as a safety net.
- **Thermal**: If the SoC throttles (> 80°C), add active cooling / a heatsink-fan. (On RPi, check temperature with `vcgencmd measure_temp`.)

### 6.3: Understanding the Two-Stage Mode (Anomaly Gate + Spark Classifier)

When `ANOMALY_ENABLED=1`:

- Most flows are classified as **Benign (Gate)** and will **skip Spark inference** to reduce load.
- Only flows flagged by the gate are sent into the PySpark model for final classification.

All flows are still stored in PostgreSQL:

- `raw_features->>'route' = 'anomaly_gate_only'` for skipped flows
- `raw_features->>'route' = 'spark_classifier'` for flows processed by Spark

---

## TROUBLESHOOTING

| Error | Cause | Solution |
|---|---|---|
| `Connection refused :9092` | Kafka not started on Mac | Run `docker compose up -d` on Mac |
| `Model not found` | Model not copied to the edge node | Run `save_model.py` + scp |
| `java: command not found` | Java not installed on the edge node | `sudo apt install default-jdk` |
| Jetson/RPi freezes / slow | Out of RAM | Check `htop`/`tegrastats`, increase swap |
| `No brokers available` | Wrong IP in .env | Check Mac IP: `ifconfig` |
