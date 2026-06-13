# Distributed deployment: 1× Mac + 2× Jetson Nano

All ML pipelines (`ml_01`–`ml_07`) and edge inference use a **Spark standalone cluster**.

```
┌─────────────────────────────────────────────────────────────┐
│  Mac                                                         │
│  • Spark Master (:7077) + Web UI (:8080)                    │
│  • Docker: Kafka, PostgreSQL, InfluxDB, Grafana             │
│  • ml_00 (CSV → parquet), save_model.py                     │
└───────────────┬─────────────────────┬───────────────────────┘
                │                     │
     ┌──────────┴──────────┐ ┌────────┴──────────┐
     │  Jetson Nano #1       │ │  Jetson Nano #2   │
     │  Spark Worker         │ │  Spark Worker     │
     │  PySpark Driver (ML)  │ │  edge classifier  │
     │  anomaly_gate (opt.)  │ │  (SOICT split)    │
     └───────────────────────┘ └───────────────────┘
```

---

## Step 0 — Configure `cluster/spark_cluster.env`

```bash
cd /path/to/Thesis_IDS
cp cluster/spark_cluster.env.example cluster/spark_cluster.env
```

Edit the following:

| Variable | Example | Notes |
|----------|---------|-------|
| `MAC_IP` | `192.168.1.101` | Mac: `ipconfig getifaddr en0` |
| `JETSON1_IP` | `192.168.1.50` | Jetson #1: `hostname -I` |
| `JETSON2_IP` | `192.168.1.102` | Jetson #2 (when available) |
| `JETSON_SSH_USER` | `bvdung` | Not always `jetson` |
| `JETSON2_ENABLED` | `0` or `1` | `0` = single-Jetson trial |
| `IDS_MAC_ROOT` | `/Users/you/Desktop/Thesis_IDS` | Project path on **Mac** |
| `IDS_RAW_DATA_DIR` | `.../ids-2017` | CSV on Mac |
| `IDS_DATA_DIR` | `.../data` | Parquet on Mac |

**Important:** Do not set `IDS_ROOT` to a Jetson path on Mac — use `IDS_MAC_ROOT` for sync.

Update Kafka Docker on Mac — `raspberry/docker-compose.yml`:

```yaml
KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka:29092,EXTERNAL://<MAC_IP>:9092
```

---

## Phase 1 — Mac + 1 Jetson (trial)

### Mac

```bash
source cluster/load_cluster_env.sh   # works in zsh and bash

./cluster/start_master_mac.sh
cd raspberry && docker compose up -d
python scripts/init_kafka_topics.py --partitions 2 --bootstrap localhost:9092
```

Spark UI: http://`<MAC_IP>`:8080

### Jetson #1

```bash
# From Mac — SSH key (one-time)
ssh-copy-id <user>@<JETSON1_IP>

# Initial code sync
rsync -avz --exclude venv --exclude .git \
  ~/Desktop/Thesis_IDS/ <user>@<JETSON1_IP>:~/Thesis_IDS/
scp cluster/spark_cluster.env <user>@<JETSON1_IP>:~/Thesis_IDS/cluster/

# On Jetson
ssh <user>@<JETSON1_IP>
cd ~/Thesis_IDS/raspberry && ./scripts/setup_jetson.sh
cd ~/Thesis_IDS
unset SPARK_HOME    # if previously set to wrong /opt/spark
source cluster/load_cluster_env.sh
./cluster/start_worker.sh
```

### Mac — sync data + ML

```bash
python ml_00_prepare_cicids2017.py   # Mac-only, one-time
./cluster/sync_workspace.sh
./cluster/check_cluster.sh           # 1 worker, parquet OK
./cluster/run_ml_remote.sh ml_01_baseline_all_features.py
```

---

## Phase 2 — Add Jetson #2

### 1. Update config (Mac)

```bash
# cluster/spark_cluster.env
export JETSON2_IP=192.168.1.XXX
export JETSON2_ENABLED=1
```

### 2. Setup Jetson #2 (same as Jetson #1)

```bash
ssh-copy-id <user>@<JETSON2_IP>
rsync -avz --exclude venv --exclude .git \
  ~/Desktop/Thesis_IDS/ <user>@<JETSON2_IP>:~/Thesis_IDS/
scp cluster/spark_cluster.env <user>@<JETSON2_IP>:~/Thesis_IDS/cluster/

ssh <user>@<JETSON2_IP>
cd ~/Thesis_IDS/raspberry && ./scripts/setup_jetson.sh
cd ~/Thesis_IDS && source cluster/load_cluster_env.sh && ./cluster/start_worker.sh
```

### 3. Mac — sync + verify

```bash
source cluster/load_cluster_env.sh
./cluster/sync_workspace.sh    # sync to Jetson #1 and #2
./cluster/check_cluster.sh     # 2 workers ALIVE
```

Spark UI should show **Alive Workers: 2**.

### 4. Edge SOICT (2 Jetsons)

| Jetson | `.env` | Role |
|--------|--------|------|
| #1 | `.env.jetson1.example` | `EDGE_NODE_ROLE=anomaly_gate` |
| #2 | `.env.jetson2.example` | `EDGE_NODE_ROLE=classifier` |

Both point Kafka/DB to `MAC_IP`. Details: [raspberry/JETSON_DISTRIBUTED.md](../raspberry/JETSON_DISTRIBUTED.md)

---

## ML dependencies on Jetson

`setup_jetson.sh` installs edge deps. ML driver also needs (auto-installed on first `run_ml_remote.sh`):

```bash
cd ~/Thesis_IDS/raspberry && source venv/bin/activate
pip install -r ../cluster/requirements_ml_driver_min.txt   # ml_01–ml_04, ml_07
pip install -r ../cluster/requirements_ml_driver.txt       # + xgboost, shap (ml_05–ml_06)
```

---

## Reproduce scripts

```bash
./cluster/reproduce_cluster.sh fair
./cluster/reproduce_cluster.sh soict
./cluster/reproduce_cluster.sh thesis
```

Or run individual scripts:

```bash
./cluster/run_ml_remote.sh ml_07_cross_method_comparison.py
```

---

## Key environment variables

| Variable | Role |
|----------|------|
| `IDS_SPARK_CLUSTER=1` | Enable cluster mode in `shared_utils.py` |
| `SPARK_MASTER` | `spark://<MAC_IP>:7077` |
| `SPARK_DRIVER_HOST` | Jetson #1 IP |
| `JETSON2_ENABLED` | `0` = skip SSH/sync/stop for Jetson #2 |
| `IDS_MAC_ROOT` | Project root on Mac (sync source) |
| `IDS_CLUSTER_DATA_DIR` | Parquet path on Jetson (`.../data`) |
| `SPARK_EXECUTOR_MEMORY` | `768m` (Jetson 4GB) |

**Mac-only** (local Spark OK): `ml_00`, `save_model.py` with `IDS_ALLOW_LOCAL_SPARK=1`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `BASH_SOURCE parameter not set` | Use `source cluster/load_cluster_env.sh` (zsh fix applied) |
| `rsync from spark://...` | Old scripts; master/worker now `unset SPARK_MASTER` |
| SSH `Permission denied` | Correct `JETSON_SSH_USER`, `ssh-copy-id`, enable `PasswordAuthentication yes` on Jetson |
| 0 workers on UI | Run `./cluster/start_worker.sh` **on Jetson**; check Mac firewall :7077 |
| `SPARK_HOME=/opt/spark` | `unset SPARK_HOME` before starting worker |
| `Spark sbin not found` | Install pyspark in venv; script uses `pyspark.__path__[0]` |
| `Java gateway exited` | `sudo apt install default-jdk`; verify `java -version` on Jetson |
| `No module named pandas` | `pip install -r cluster/requirements_ml_driver_min.txt` |
| pip `IncompleteRead` on Jetson | Install packages one-by-one; use `_min.txt` first |
| `Missing parquet on Jetson` | `./cluster/sync_workspace.sh`; verify `IDS_MAC_ROOT` on Mac |
| Kafka `NoBrokersAvailable` | Wait 30s after `docker compose up`; use `--bootstrap localhost:9092` on Mac |
| OOM on Jetson | Reduce `SPARK_EXECUTOR_MEMORY`; increase swap |

---

## Code layout

| File | Role |
|------|------|
| `shared_utils.py` | `create_spark_session()` — cluster + auto JAVA_HOME |
| `cluster/load_cluster_env.sh` | Load config (zsh + bash) |
| `cluster/sync_workspace.sh` | rsync Mac → Jetson(s) |
| `cluster/run_ml_remote.sh` | SSH driver on Jetson #1 |
| `cluster/start_master_mac.sh` | Spark master on Mac |
| `cluster/start_worker.sh` | Spark worker on Jetson |
| `cluster/stop_cluster.sh` | Stop master + workers |
| `cluster/check_cluster.sh` | Health check |
