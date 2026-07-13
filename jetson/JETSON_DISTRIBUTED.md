# Distributed IDS on 2× Jetson Orin Nano Super Developer Kit (8 GB)

Guide for running the edge IDS on **two Jetson Orin Nano Super Developer Kits (8 GB RAM, 256 GB NVMe)** connected to the Kafka / PostgreSQL / InfluxDB infrastructure on the Mac.

**Distributed Spark ML training:** see [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md).

---

## Lab IPs (reference)

| Node | IP | Notes |
|------|-----|-------|
| Mac | `192.168.1.165` | Spark Master, Docker — `MAC_IP` |
| Jetson #1 | `192.168.1.50` | Driver + Worker, `anomaly_gate` |
| Jetson #2 | `192.168.1.205` | Worker, `classifier` |

Check the Mac IP with `ipconfig getifaddr en0` — it must match `MAC_IP` in `cluster/spark_cluster.env` and `KAFKA_ADVERTISED_LISTENERS` in `docker-compose.yml`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Mac (host)                             │
│  Docker: Kafka, PostgreSQL, InfluxDB, Grafana                   │
│  data_sender.py → topic: ids-network-flow                       │
└───────────────┬───────────────────────────────┬─────────────────┘
                │                               │
     ┌──────────┴──────────┐         ┌──────────┴──────────┐
     │  Jetson Orin Nano #1 │         │  Jetson Orin Nano #2 │
     │  (anomaly_gate)      │────────▶│  (classifier)        │
     │  sklearn AE filter   │  Kafka  │  PySpark model       │
     └──────────────────────┘         └──────────────────────┘
              ids-suspicious-flow
```

The system supports **3 distributed modes** (all on Mac + 2 Jetson):

| Mode | Description | When to use |
|------|-------------|-------------|
| **A. Pipeline split** | Jetson #1 = anomaly gate, Jetson #2 = classifier | Offload Spark; default / recommended |
| **B. Horizontal scaling** | Both Jetsons run the full pipeline in one consumer group | Maximize throughput |
| **C. Spark cluster** | Mac = Spark master, both Jetsons = workers | Distributed Spark inference / training |

---

## Hardware & network requirements

- 2× Jetson Orin Nano Super Developer Kit (**8 GB RAM**, **256 GB NVMe** recommended)
- Mac/PC on the **same LAN** as both Jetsons (`192.168.1.x`)
- Stable 5 V / 4 A supply per Jetson
- 4 GB swap (optional on 8 GB — configured by `setup_jetson.sh`)

---

## Step 1 — Infrastructure on the Mac

**Spark cluster (distributed ML training):** see [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md).

```bash
cd jetson/
docker compose up -d

# Create Kafka topics with ≥ 2 partitions
source venv/bin/activate   # or: pip install kafka-python
python scripts/init_kafka_topics.py --partitions 2
```

Set the Mac IP (`192.168.1.165`) in `docker-compose.yml` (line `KAFKA_ADVERTISED_LISTENERS`).

Export the models (if not already done):

```bash
python scripts/save_model.py
# Optional anomaly gate:
cd .. && python ml_08_anomaly_gate_autoencoder.py
```

---

## Step 2 — Provision each Jetson Orin Nano Super

On **both** Jetsons:

```bash
scp -r jetson/        <user>@<jetson-ip>:~/Thesis_IDS/jetson
scp -r jetson/model/* <user>@<jetson-ip>:~/Thesis_IDS/jetson/model/

ssh <user>@<jetson-ip>
cd ~/Thesis_IDS/jetson
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

---

## Step 3 — Choose a deployment mode

### Mode A — Pipeline split (recommended)

**Jetson #1** — anomaly gate (lightweight filter, no Spark needed):

```bash
cp .env.jetson1.example .env
nano .env   # set the Mac IP
source venv/bin/activate
EDGE_NODE_ID=jetson-nano-1 EDGE_NODE_ROLE=anomaly_gate ALERT_ENABLED=0 \
  python edge/kafka_consumer.py
```

**Jetson #2** — PySpark classifier:

```bash
cp .env.jetson2.example .env
nano .env   # set the Mac IP
source venv/bin/activate
EDGE_NODE_ID=jetson-nano-2 EDGE_NODE_ROLE=classifier ALERT_ENABLED=1 \
  python edge/kafka_consumer.py
```

Key environment variables:

| Jetson | EDGE_NODE_ID | EDGE_NODE_ROLE | ALERT_ENABLED |
|--------|--------------|----------------|---------------|
| #1 | `jetson-nano-1` | `anomaly_gate` | `0` |
| #2 | `jetson-nano-2` | `classifier` | `1` |

Data flow:
1. Mac sends flows → `ids-network-flow`
2. Jetson #1 scores them with the autoencoder and forwards suspicious flows → `ids-suspicious-flow`
3. Jetson #2 classifies with PySpark, stores results, and sends alerts

---

### Mode B — Horizontal scaling

Both Jetsons run the full pipeline in the **same** `KAFKA_GROUP_ID`:

```bash
cp .env.jetson-horizontal.example .env
# Jetson #1: EDGE_NODE_ID=jetson-nano-1
# Jetson #2: EDGE_NODE_ID=jetson-nano-2, ALERT_ENABLED=0
EDGE_NODE_ROLE=full python edge/kafka_consumer.py
```

Kafka splits the partitions across the two consumers → roughly double the throughput.

---

### Mode C — Spark cluster (ML training)

**Use the Mac as the Spark master**, both Jetsons as workers (the Jetsons are not masters).

Full guide: [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md)

```bash
# Mac
./cluster/start_master_mac.sh

# Each Jetson
./cluster/start_worker.sh

# Mac — train
./cluster/run_ml_remote.sh ml_01_baseline_all_features.py
./cluster/pull_results.sh
```

---

## Step 4 — Send test data

On the Mac:

```bash
cd jetson/
python sender/data_sender.py --csv /path/to/CICIDS2017.csv --rate 100
```

---

## Step 5 — Monitoring

- **Grafana:** `http://<mac-ip>:3000` — metrics tagged by `host = jetson-nano-1/2`
- **PostgreSQL:** `node_id` column in the `predictions` and `alerts` tables
- **InfluxDB:** tag `host` = `EDGE_NODE_ID`

---

## Code layout

```
jetson/edge/
├── kafka_consumer.py      # entry point
├── role_pipelines.py      # full | anomaly_gate | classifier
├── pipeline_base.py       # shared storage / monitor / alert
├── kafka_forwarder.py     # forward suspicious flows
└── ...
```

Configuration variables in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_NODE_ID` | `edge-node-1` | Unique ID per Jetson |
| `EDGE_NODE_ROLE` | `full` | `full` / `anomaly_gate` / `classifier` |
| `KAFKA_SUSPICIOUS_TOPIC` | `ids-suspicious-flow` | Topic for pipeline split |
| `ALERT_ENABLED` | `1` | Disable on the secondary node to avoid duplicate alerts |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Only one Jetson receives messages | Topic has 1 partition | `python scripts/init_kafka_topics.py --partitions 2` |
| Jetson #2 has no data | Gate not running / wrong topic | Check Jetson #1 log for `Forwarded: …` |
| PySpark OOM | Not enough RAM | Lower `SPARK_EXECUTOR_MEMORY`; add swap; use Mode A |
| Spark cluster won't connect | Firewall on port 7077 | `sudo ufw allow 7077` on Jetson #1 |
| High temperature | No cooling on the Jetson | `sudo jetson_clocks`, add a fan |

---

## Notes for the thesis

- **Mode A** clearly demonstrates the distributed pipeline-split architecture (edge computing).
- Compare 1-node vs 2-node latency/throughput with `scripts/benchmark.py`.
- Filter Grafana panels by the `host` tag to visualize each Jetson.
- Use the PostgreSQL `node_id` column to analyze load distribution across nodes.

### Measurement notes (latency & energy)

- **End-to-end latency** (send → verdict) is computed from the producer's
  `_timestamp`; for the absolute value to be meaningful the **sender host and
  both Jetson nodes must share an NTP-synced clock** (e.g. `sudo timedatectl
  set-ntp true` on all hosts). Relative comparisons across modes hold even
  without NTP. Each node also reports inference-only latency separately.
- **Latency percentiles** (p50/p95/p99) are computed by each node from its raw
  per-flow samples; the cluster p95 is the **worst node's** p95 — never a
  percentile of per-host mean latencies.
- **Raw latency logs (required for the paper numbers):** set
  `RAW_LATENCY_LOG=~/ids_raw_latency_$(hostname).csv` on each node **before**
  starting the pipelines. Window-level p95s pushed to InfluxDB are for
  dashboards only; the *run-level* p95 is computed by the orchestrator from
  these raw CSVs restricted to the exact load window
  (`benchmark_distributed.py collect --raw-latency-glob '.../ids_raw_latency_*.csv'`).
  Without them, `collect` falls back to an InfluxDB approximation and labels it
  `DO_NOT_PUBLISH`.
- **Per-node energy during a distributed run:** on each Jetson run
  `./papers/soict2026/run_benchmarks.sh node-power` (measures a 30 s idle
  baseline first, then samples tegrastats through the load window). Mode A's
  paper figure is the **sum of both nodes' active energy** ÷ classified flows.
- **Repetitions & warmup:** `run` defaults to 5 repeats with real warmup
  traffic excluded from the measured window; `merge` writes
  `summary_mean_std.csv` per mode. Do a load sweep (`BENCHMARK_RATE=50|100|200`)
  so throughput/latency are reported relative to saturation.
- **Pipeline throughput** is counted as **final verdicts/s from Postgres**
  within the load window — never the sum of per-node rates, which double-counts
  forwarded flows in Mode A.
- **Energy** is reported both raw and **idle-subtracted (active)** per node via
  `tegrastats`. For pipeline-split (Mode A) the comparable figure is the **sum
  of both nodes'** active energy (gate on Jetson #1 + classifier on Jetson #2),
  divided by the number of classified flows.
- **Inference-engine baseline:** `scripts/benchmark_engines.py` runs the same
  RandomForest / SHAP Top-30 model through scikit-learn and ONNX Runtime
  (`pip install skl2onnx onnxruntime` to enable ONNX) and reports the same
  metrics as `benchmark.py` (Spark). This quantifies the JVM/Spark overhead;
  Spark is kept for train-serve consistency, sklearn/ONNX motivate a future
  export path. Example: `python scripts/benchmark_engines.py --samples 5000`.
