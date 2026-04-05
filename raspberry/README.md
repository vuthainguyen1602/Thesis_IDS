# IDS Edge Deployment — Detailed Guide

This guide covers the deployment of the **PySpark-based IDS** onto a **Raspberry Pi 4B (8GB)** using a **Split Deployment** architecture.

---

## ⚡ Quick Start Checklist

If you are already familiar with the setup, use these commands in order:

1. **Mac**: `cd raspberry && docker compose up -d` (Start Infra)
2. **Mac**: `python scripts/save_model.py` (Export trained model)
3. **RPi**: `./scripts/setup_raspberry.sh` (First time only)
4. **RPi**: `scp -r mac_user@mac_ip:~/path/to/model ~/raspberry/model/`
5. **RPi**: `python edge/kafka_consumer.py` (Start Detection)
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
│                 RASPBERRY PI 4B                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Kafka Consumer → Preprocessor → PySpark Model   │   │
│  │  → Performance Monitor → Alert (Email/Slack)      │   │
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

### Step 1.4: Find the Mac's IP Address

```bash
# Get the Mac's LAN IP address
ifconfig | grep "inet " | grep -v 127.0.0.1

# Example output: inet 192.168.1.100 netmask 0xffffff00
# Note this IP (e.g. 192.168.1.100)
```

---

## PART 2: SETUP ON RASPBERRY PI

### Step 2.1: Prepare Raspberry Pi OS

```bash
# Install Raspberry Pi OS 64-bit using Raspberry Pi Imager
# Download: https://www.raspberrypi.com/software/
# Select: Raspberry Pi OS (64-bit) Lite or Desktop
# Write to a 32GB SD Card

# Insert the SD card, power on the RPi, connect to the same WiFi/Ethernet network as the Mac
```

### Step 2.2: SSH into the Raspberry Pi

```bash
# From Mac, SSH into the RPi
ssh pi@<rpi-ip>
# Default password: raspberry (change it immediately)

# Or if using the Desktop version, open Terminal on the RPi
```

### Step 2.3: Clone the Project to the RPi

```bash
# On RPi
cd ~
git clone <repo-url>
cd raspberry/

# OR: Copy directly from Mac to RPi
# On Mac, run:
# scp -r /Users/thainguyenvu/Desktop/Thesis_IDS/raspberry pi@<rpi-ip>:~/raspberry
```

### Step 2.4: Run the Automated Setup Script

```bash
# On RPi
cd ~/raspberry
chmod +x scripts/setup_raspberry.sh
./scripts/setup_raspberry.sh

# The script will automatically:
# [1/6] Update system packages
# [2/6] Install Python3 + pip + dev tools
# [3/6] Ask about Docker → select N (Docker runs on Mac)
# [4/6] Install Java JDK (required for PySpark)
# [5/6] Create Python venv + install dependencies (pyspark, kafka-python, ...)
# [6/6] Create .env file from template

# Estimated time: ~10–15 minutes (Java + PySpark downloads take a while)
```

### Step 2.5: Configure .env

```bash
# On RPi, edit the .env file
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
# On RPi
scp -r <mac-user>@<mac-ip>:/Users/thainguyenvu/Desktop/Thesis_IDS/raspberry/model/ ~/raspberry/model/

# Verify
ls -la ~/raspberry/model/
# Should contain:
#   ids_pipeline_model/    (PySpark model directory)
#   feature_columns.json   (JSON file)
```

### Step 2.7: Test Connectivity

```bash
# On RPi, verify the Mac is reachable
ping <mac-ip> -c 3

# Test Kafka
nc -zv <mac-ip> 9092     # Expected: Connection to ... succeeded!

# Test PostgreSQL
nc -zv <mac-ip> 5432     # Expected: Connection to ... succeeded!
```

---

## PART 3: RUNNING THE SYSTEM

### Step 3.1: Start IDS on the RPi

```bash
# On RPi - Terminal 1
cd ~/raspberry
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

# Note: Spark Session initialization takes ~20–30 seconds on RPi
```

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

### Step 3.3: Monitor Results on the RPi

```
# RPi terminal will display:
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
# - RPi CPU / Memory usage
# - RPi CPU temperature
# - Recent alerts table
```

---

## PART 4: ALERTING CONFIGURATION (OPTIONAL)

The IDS supports real-time notifications via Email (Mailtrap) and Slack. These are triggered automatically when an attack is detected.

### 4.1: Setup Mailtrap (Email)
Mailtrap allows you to test SMTP email delivery without sending real emails to your personal inbox.

1.  **Register**: Create a free account at [mailtrap.io](https://mailtrap.io).
2.  **Get Credentials**: Go to **Inboxes** → **My Inbox** → **SMTP Settings**.
3.  **Configure `.env`**: Copy the `Username` and `Password` to your `.env` file on the Raspberry Pi:
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
# On RPi: Ctrl+C to stop the pipeline
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
To monitor the Raspberry Pi's resource usage in real-time, use `htop` via SSH:

```bash
# On RPi
htop
```
Watch for:
- **CPU Usage**: PySpark will use multiple cores during batch inference.
- **Memory**: Ensure the resident set size (RSS) stays within the RPi's physical RAM (leave ~1GB for OS).
- **Thermal**: If the CPU throttles (> 80°C), consider adding a fan or heat sink.

---

## TROUBLESHOOTING

| Error | Cause | Solution |
|---|---|---|
| `Connection refused :9092` | Kafka not started on Mac | Run `docker compose up -d` on Mac |
| `Model not found` | Model not copied to RPi | Run `save_model.py` + scp |
| `java: command not found` | Java not installed on RPi | `sudo apt install default-jdk` |
| RPi freezes / slow | Out of RAM | Check `htop`, increase swap |
| `No brokers available` | Wrong IP in .env | Check Mac IP: `ifconfig` |
