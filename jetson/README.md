# Edge IDS — Distributed Deployment (Mac + 2× Jetson Orin Nano Super)

Real-time intrusion detection on **two NVIDIA Jetson Orin Nano Super Developer Kits (8 GB RAM, 256 GB NVMe)** using a **split-deployment** architecture. Central infrastructure (Kafka, PostgreSQL, InfluxDB, Grafana) runs in Docker on the **Mac**; inference runs on the **two Jetson edge nodes**.

> This is the only supported edge configuration: **1 Mac + 2 Jetson**. Distributed Spark training on the same cluster is documented in [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md). The full mode reference (pipeline split / horizontal scaling / Spark cluster) is in [JETSON_DISTRIBUTED.md](JETSON_DISTRIBUTED.md).

---

## Architecture (pipeline split — default)

```
┌───────────────────────────────────────────────────────────────┐
│                         Mac (host)                            │
│  Docker: Kafka  ·  PostgreSQL  ·  InfluxDB  ·  Grafana :3000  │
│  sender/data_sender.py  →  topic: ids-network-flow           │
└───────────────┬───────────────────────────────┬──────────────┘
                │            same LAN            │
     ┌──────────┴──────────┐         ┌──────────┴───────────┐
     │  Jetson #1           │  Kafka  │  Jetson #2            │
     │  EDGE_NODE_ROLE=     │ ──────▶ │  EDGE_NODE_ROLE=      │
     │   anomaly_gate       │  ids-   │   classifier          │
     │  sklearn AE filter   │ suspic- │  PySpark PipelineModel│
     │  forwards suspicious │ ious-   │  Benign / Attack +    │
     │  flows only          │ flow    │  store + alert        │
     └──────────────────────┘         └───────────────────────┘
```

**Flow:** Mac streams network flows to `ids-network-flow` → Jetson #1 scores each flow with a lightweight autoencoder and forwards only *suspicious* flows to `ids-suspicious-flow` → Jetson #2 classifies them with the exported PySpark model, writes results to PostgreSQL, and raises alerts.

---

## Roles (set per node via `EDGE_NODE_ROLE`)

| Node | `EDGE_NODE_ID` | `EDGE_NODE_ROLE` | `ALERT_ENABLED` | Function |
|------|----------------|------------------|-----------------|----------|
| Jetson #1 | `jetson-nano-1` | `anomaly_gate` | `0` | Autoencoder gate; forwards suspicious flows |
| Jetson #2 | `jetson-nano-2` | `classifier` | `1` | PySpark classifier on suspicious flows |

`EDGE_NODE_ROLE=full` (single node running the whole pipeline) is also supported by the code and is used by **Mode B – horizontal scaling** (both Jetsons in one Kafka consumer group). See [JETSON_DISTRIBUTED.md](JETSON_DISTRIBUTED.md).

---

## Part 1 — Mac (one-time)

### 1.1 Start the infrastructure (Docker)

```bash
cd jetson/
docker compose up -d
docker compose ps          # kafka, postgres, influxdb, grafana, zookeeper → running
```

Set the Mac LAN IP in `docker-compose.yml` (line `KAFKA_ADVERTISED_LISTENERS`) so the Jetsons can reach Kafka:

```bash
ipconfig getifaddr en0     # e.g. 192.168.1.165 — must match MAC_IP in cluster/spark_cluster.env
```

### 1.2 Create Kafka topics (≥ 2 partitions)

```bash
source venv/bin/activate                 # or: pip install kafka-python
python scripts/init_kafka_topics.py --partitions 2 --bootstrap localhost:9092
```

### 1.3 Export the models

```bash
# Classifier (PySpark PipelineModel) → jetson/model/
python scripts/save_model.py
#   → model/ids_pipeline_model/      (PySpark PipelineModel)
#   → model/feature_columns.json     (SHAP Top-30 feature list)

# Anomaly gate (sklearn autoencoder) → jetson/model/
cd .. && python ml_08_anomaly_gate_autoencoder.py
#   → jetson/model/anomaly_autoencoder.pkl
#   → jetson/model/anomaly_scaler.pkl
#   → jetson/model/anomaly_threshold.json
```

---

## Part 2 — Each Jetson (one-time)

Run on **both** Jetsons (only the SSH target differs):

```bash
# From Mac — copy code + models to the Jetson
scp -r jetson/        <user>@<jetson-ip>:~/Thesis_IDS/jetson
scp -r jetson/model/* <user>@<jetson-ip>:~/Thesis_IDS/jetson/model/

# On the Jetson — automated setup
ssh <user>@<jetson-ip>
cd ~/Thesis_IDS/jetson
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

`setup_jetson.sh` installs Python + Java (`default-jdk`), creates the venv (pyspark, kafka-python, scikit-learn, …), adds a 4 GB safety swap, and writes a `.env` template.

### Configure `.env` (point to the Mac)

```bash
# Jetson #1
cp .env.jetson1.example .env
# Jetson #2
cp .env.jetson2.example .env

nano .env   # set the Mac IP in:
#   KAFKA_BOOTSTRAP_SERVERS=<mac-ip>:9092
#   POSTGRES_HOST=<mac-ip>
#   INFLUXDB_URL=http://<mac-ip>:8086
```

The Jetsons **do not run Docker** — they connect to the Mac's services over the LAN.

---

## Part 3 — Run (Mode A: pipeline split)

**Jetson #1 — anomaly gate** (no Spark required):

```bash
cd ~/Thesis_IDS/jetson && source venv/bin/activate
EDGE_NODE_ID=jetson-nano-1 EDGE_NODE_ROLE=anomaly_gate ALERT_ENABLED=0 \
  python edge/kafka_consumer.py
```

**Jetson #2 — PySpark classifier:**

```bash
cd ~/Thesis_IDS/jetson && source venv/bin/activate
EDGE_NODE_ID=jetson-nano-2 EDGE_NODE_ROLE=classifier ALERT_ENABLED=1 \
  python edge/kafka_consumer.py
```

(These variables can instead be placed in each node's `.env`.) Spark session init takes ~10–15 s on the Jetson Orin Nano Super.

**Mac — stream test data** (CICIDS2017 CSV):

```bash
cd jetson/
python sender/data_sender.py --csv /path/to/CICIDS2017.csv --rate 100
```

---

## Part 4 — Monitoring

- **Grafana:** `http://<mac-ip>:3000` (admin / admin). Import `dashboard/grafana_dashboard.json`. Panels: throughput, detected attacks, latency, per-node CPU/RAM/temperature (tagged by `EDGE_NODE_ID`), prediction routes (`anomaly_gate_only` vs `spark_classifier`), recent alerts.
- **PostgreSQL:** `predictions` and `alerts` tables carry a `node_id` column for per-node load analysis.
- **InfluxDB:** metrics tagged with `host = EDGE_NODE_ID`.

### Energy per inference

`edge/power_monitor.py` samples `tegrastats` (board power rails) while the pipeline runs and integrates it into energy-per-inference (mJ). This feeds the SOICT edge benchmark (`scripts/benchmark.py`).

---

## Part 5 — Alerting (optional)

Alerts fire automatically on the node with `ALERT_ENABLED=1` (Jetson #2) when an attack is detected.

**Email (Mailtrap)** — add to `.env`:

```env
SMTP_HOST=sandbox.smtp.mailtrap.io
SMTP_PORT=2525
SMTP_USER=<username>
SMTP_PASSWORD=<password>
ALERT_EMAIL_TO=you@example.com
```

**Slack** — create an Incoming Webhook and add to `.env`:

```env
WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../X...
```

---

## Part 6 — Stop

```bash
# Each Jetson: Ctrl+C to stop the pipeline (prints final statistics)
# Mac: Ctrl+C to stop the sender, then:
docker compose down
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused :9092` | Kafka not running / wrong IP | `docker compose up -d` on Mac; check `KAFKA_BOOTSTRAP_SERVERS` |
| Jetson #2 receives no data | Gate not running or topic missing | Check Jetson #1 log for `Forwarded: …`; recreate topics with `--partitions 2` |
| `Model not found` | Model not copied to Jetson | Re-run `save_model.py` on Mac, then `scp` `model/` |
| `java: command not found` | JDK missing on Jetson | `sudo apt install -y default-jdk` |
| PySpark OOM on Jetson #2 | 8 GB RAM exhausted | Lower `SPARK_EXECUTOR_MEMORY`; ensure swap is enabled (`setup_jetson.sh`) |
| SoC throttling (> 80 °C) | No active cooling | Attach a heatsink-fan; `sudo jetson_clocks` |

---

## Code layout

```
jetson/
├── docker-compose.yml          # Mac infra: Kafka, Postgres, InfluxDB, Grafana
├── config.py                   # EDGE_NODE_ROLE, topics, DB/Kafka endpoints
├── sender/data_sender.py       # CSV → Kafka (Mac)
├── edge/
│   ├── kafka_consumer.py       # entry point (reads EDGE_NODE_ROLE)
│   ├── role_pipelines.py       # full | anomaly_gate | classifier
│   ├── kafka_forwarder.py      # forward suspicious flows (gate → classifier)
│   ├── anomaly_scorer.py       # sklearn autoencoder gate
│   ├── prediction_engine.py    # PySpark PipelineModel inference
│   ├── performance_monitor.py  # CPU/RAM/temp/throughput → InfluxDB
│   └── power_monitor.py        # tegrastats energy sampling
├── scripts/
│   ├── setup_jetson.sh         # one-time Jetson provisioning
│   ├── save_model.py           # export PySpark model
│   ├── init_kafka_topics.py    # create topics (≥ 2 partitions)
│   └── benchmark.py            # edge latency/throughput/energy benchmark
└── model/                      # exported models (copied to each Jetson)
```

Key `config.py` variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_NODE_ID` | `edge-node-1` | Unique ID per Jetson (metric/DB tag) |
| `EDGE_NODE_ROLE` | `full` | `full` / `anomaly_gate` / `classifier` |
| `KAFKA_SUSPICIOUS_TOPIC` | `ids-suspicious-flow` | Topic between gate and classifier |
| `ALERT_ENABLED` | `1` | Set `0` on the gate node to avoid duplicate alerts |
| `ANOMALY_ENABLED` | `0` | Enable the autoencoder gate inside a `full` pipeline |
