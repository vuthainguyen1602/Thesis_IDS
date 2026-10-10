# Distributed IDS on 2× Jetson Orin Nano Super Developer Kit (8 GB)

Guide for running the edge IDS on **two Jetson Orin Nano Super Developer Kits (8 GB RAM, 256 GB NVMe)** connected to the Kafka / PostgreSQL / InfluxDB infrastructure on the Mac.

**Distributed Spark ML training:** see [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md).

---

## Lab IPs (reference)

| Node | IP | Notes |
|------|-----|-------|
| Mac | `192.168.1.165` | Spark Master, Docker — `MAC_IP` |
| Jetson #1 | `192.168.1.50` | Driver + Worker, `anomaly_gate` |
| Jetson #2 | `192.168.1.204` | Worker, `classifier` |

Check the Mac IP with `ipconfig getifaddr en0` — it must match `MAC_IP` in `cluster/spark_cluster.env` and `KAFKA_ADVERTISED_LISTENERS` in `docker-compose.yml`.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                             Mac (host)                            │
│  Docker: Kafka, PostgreSQL, InfluxDB, Grafana                     │
│  data_sender.py → topic: ids-network-flow                         │
└───────────────┬───────────────────────────────────┬───────────────┘
                │                                   │
  ┌─────────────┴─────────────┐       ┌─────────────┴─────────────┐
  │ Jetson Orin Nano Super #1 │       │ Jetson Orin Nano Super #2 │
  │ (anomaly_gate)            │──────▶│ (classifier)              │
  │ sklearn AE filter         │ Kafka │ PySpark model             │
  └───────────────────────────┘       └───────────────────────────┘
                         ids-suspicious-flow
```

The system supports **3 distributed modes** (all on Mac + 2 Jetson):

| Mode | Description | When to use |
|------|-------------|-------------|
| **A. Pipeline split** | Jetson #1 = anomaly gate, Jetson #2 = classifier | Offload Spark; default / recommended |
| **B. Horizontal scaling** | Both Jetsons run the full pipeline in one consumer group | Redundancy; adds little capacity (3.3 vs 2.5 flows/s on one node, gate off) |
| **C. Spark cluster** | Mac = Spark master, both Jetsons = workers | Distributed Spark inference / training |

---

## Hardware & network requirements

- 2× Jetson Orin Nano Super Developer Kit (**8 GB RAM**, **256 GB NVMe** recommended)
- Mac/PC on the **same LAN** as both Jetsons (`192.168.1.x`)
- Stable 5 V / 4 A supply per Jetson
- 4 GB swap (optional on 8 GB — configured by `setup_jetson.sh`). For Spark
  *training* on the same boards use `cluster/setup_swap_jetson.sh` instead: 8 GB on
  the NVMe, zram disabled.

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

Kafka splits the partitions across the two consumers, but each board still scores its share with PySpark, so capacity barely moves: 3.3 flows/s against 2.5 on one node (gate off, 2026-09-30 re-measurement in `papers/soict2026/results/remeasure_20260930/`). Use it for redundancy; for capacity use Mode A.

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
- Compare modes with the capacity sweep, `./papers/soict2026/run_capacity_sweep.sh`
  (run on the Mac; see the measurement notes below). `scripts/benchmark.py` is a
  single-node micro-benchmark and cannot compare modes.
- Filter Grafana panels by the `host` tag to visualize each Jetson.
- Use the PostgreSQL `node_id` column to analyze load distribution across nodes.

### Measurement notes (capacity, latency & energy)

The numbers in the SOICT paper and the thesis come from the queue-aware
re-measurement of 2026-09-29/30 (`papers/soict2026/results/remeasure_20260930/`,
whose README lists what was wrong with the earlier method). The older files in
`papers/soict2026/results/benchmarks/` (62.5 verdicts/s, 95.8% gate skip) are
superseded.

- **Follow every flow sent, not the verdicts written in the window.** Each
  verdict stores the flow's send timestamp and replay row.
  `papers/soict2026/analyze_runs.py` uses them to follow every flow sent in the
  load window to its verdict, wherever it lands. It reports per repeat: flows
  sent and completed, send-to-verdict p50/p95 *with queueing included*,
  `drain_s` (last verdict minus window end), sustained verdict rate, gate-skip
  ratio and live attack recall. Counting verdicts written inside the window
  hides the backlog of a stage that cannot keep up and inflates the skip ratio.
- **Sustained capacity** is the highest offered rate at which every repeat
  drained within 10 s of the window's end (about one PySpark batch on a Jetson).
  `papers/soict2026/run_capacity_sweep.sh` raises the rate per mode (45 s load,
  15 s warm-up, 2 repeats) and stops at the first rate that fails;
  `RATES_<mode>="..."` overrides a mode's rate list. `spark_cluster` needs the
  Spark master and workers up, so it is not in the default `MODES`.
- **Fresh Kafka consumer group per run.** `run_dist_bench.sh` tags the group
  IDs with the run time, so no run consumes flows a previous run left unread.
- **The sender runs below its target** (about 80 flows/s when asked for 100).
  Report the measured `send_rps`, not the requested rate.
- **Clocks:** end-to-end latency compares the Mac's send time with a Jetson's
  verdict time, so all three hosts must be NTP-synced (`sudo timedatectl
  set-ntp true`). Record `chronyc tracking` (or `timedatectl timesync-status`)
  on each board before a sweep, together with a ping RTT to the Mac.
- **Energy per verdict** is measured in a separate window at the sustained
  rate: `run_dist_bench.sh <mode>` without `NO_ENERGY=1` runs `node-power` on
  the active board(s) (30 s idle baseline, then `tegrastats` through the load)
  and writes `power_<node>_<ts>.json`. J/verdict is the summed average
  total-board power of the participating boards divided by the verdict rate.
  `energy_at_sustained_rate.csv` collects one such run per mode by hand; no
  script writes it. At these loads active (idle-subtracted) power rises by only
  about 1.5 W, so the table reports total-board energy.
- **Results at the sustained rate (2026-09-30):**

  | Mode | Boards | Sustained (flows/s) | e2e p50 / p95 (s) | J / verdict |
  |---|---|---|---|---|
  | Single node, gate off | 1 | 2.5 | 5.9 / 11.3 | 3.04 |
  | Single node + gate | 1 | 1.9 (2.4–2.5 at 3/s, one repeat drained in 10.6 s) | 3.8 / 7.8 | 3.04 |
  | B: Horizontal, gate off | 2 | 3.3 | 4.1 / 8.2 | 3.20 |
  | C: Spark cluster | 2 | ≥ 8.7 (≈9–17) | 0.7 / 5.2 | 1.15 |
  | A: Pipeline split | 2 | 22.4 | 1.0 / 7.1 | 0.60 |

  The gate skips about 89% of flows; live attack recall is 69.6% in split mode
  against 100% with the gate off. `papers/soict2026/plot_capacity.py` draws
  `edge_capacity.png` from the sweep CSVs and the energy CSV.
- **Inference-engine baseline:** `scripts/benchmark_same_forest.py` serves the
  *deployed* forest through the edge's own Spark, NumPy and ONNX engines and
  checks that they agree prediction by prediction
  (`python scripts/benchmark_same_forest.py --samples 1000 --batch-size 10`).
  On Jetson #2 at batch 10: PySpark 1.4 flows/s, NumPy 54.8, ONNX Runtime
  10,092, identical predictions. `scripts/benchmark_engines.py` trains its own
  scikit-learn forest, so it does not measure the deployed model.
