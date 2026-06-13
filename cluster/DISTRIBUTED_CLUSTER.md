# Distributed deployment: 1× Mac + 2× Jetson Nano Super Kit (8GB)

All ML pipelines (`ml_01`–`ml_07`) and edge inference use a **Spark standalone cluster**.

**Mac = Spark Master + Docker (no training).**  
**2× Jetson = Spark Workers (executors).**  
**Jetson #1 = PySpark driver** — writes all ML output.

---

## Lab setup (reference IPs)

| Node | IP | Role |
|------|-----|------|
| **Mac** | `192.168.1.165` | Spark Master `:7077`, Web UI `:8080`, Docker (Kafka `:9092`, PostgreSQL, InfluxDB) |
| **Jetson #1** | `192.168.1.50` | Spark Worker + **PySpark driver**, `anomaly_gate` edge |
| **Jetson #2** | `192.168.1.205` | Spark Worker, `classifier` edge |

Verify Mac IP before editing config:

```bash
ipconfig getifaddr en0    # must match MAC_IP in spark_cluster.env
```

Spark UI (when cluster is up): http://192.168.1.165:8080 — expect **Alive Workers: 2**, **8 cores**, **8 GiB**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Mac 192.168.1.165                                                │
│  • Spark Master (:7077) + Web UI (:8080)                         │
│  • Docker: Kafka :9092, PostgreSQL, InfluxDB, Grafana            │
│  • ml_00 (CSV → parquet), sync_workspace.sh, pull_results.sh     │
└───────────────┬──────────────────────────────┬───────────────────┘
                │ spark:// + rsync              │
     ┌──────────┴──────────┐         ┌───────────┴──────────┐
     │  Jetson #1           │         │  Jetson #2           │
     │  192.168.1.50        │         │  192.168.1.205       │
     │  Worker + Driver     │         │  Worker only         │
     │  results/ + model/   │         │  edge classifier     │
     │  anomaly_gate        │         │                      │
     └─────────────────────┘         └──────────────────────┘
              │ ids-suspicious-flow (Kafka on Mac)
              └──────────────────────────────────────▶ Jetson #2
```

**Training flow:** Mac SSH → Jetson #1 driver → Spark Master → executors on **both** Jetsons.  
**Results:** saved on **Jetson #1 only** → `./cluster/pull_results.sh` to copy back to Mac.

---

## Step 0 — Configure `cluster/spark_cluster.env`

```bash
cd /path/to/Thesis_IDS
cp cluster/spark_cluster.env.example cluster/spark_cluster.env
```

Edit the following:

| Variable | Example | Notes |
|----------|---------|-------|
| `MAC_IP` | `192.168.1.165` | Mac: `ipconfig getifaddr en0` |
| `JETSON1_IP` | `192.168.1.50` | Jetson #1: `hostname -I` |
| `JETSON2_IP` | `192.168.1.205` | Jetson #2 |
| `JETSON_SSH_USER` | `bvdung` | Not always `jetson` |
| `JETSON2_ENABLED` | `1` | `0` = single-Jetson trial |
| `IDS_MAC_ROOT` | `/Users/you/Desktop/Thesis_IDS` | Project path on **Mac** |
| `IDS_RAW_DATA_DIR` | `.../ids-2017` | CSV on Mac |
| `IDS_DATA_DIR` | `.../data` | Parquet on Mac |

**Important:** Do not set `IDS_ROOT` to a Jetson path on Mac — use `IDS_MAC_ROOT` for sync.

Update Kafka Docker on Mac — `raspberry/docker-compose.yml`:

```yaml
KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka:29092,EXTERNAL://192.168.1.165:9092
```

After changing `MAC_IP`, restart master, workers, and `docker compose up -d`.

---

## Phase 1 — Mac + 1 Jetson (trial)

Set `JETSON2_ENABLED=0` in `spark_cluster.env`.

### Mac

```bash
source cluster/load_cluster_env.sh

./cluster/start_master_mac.sh
cd raspberry && docker compose up -d
python scripts/init_kafka_topics.py --partitions 2 --bootstrap localhost:9092
```

### Jetson #1

```bash
# From Mac — SSH key (one-time)
ssh-copy-id bvdung@192.168.1.50

scp cluster/spark_cluster.env bvdung@192.168.1.50:~/Thesis_IDS/cluster/

# On Jetson
ssh bvdung@192.168.1.50
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
./cluster/pull_results.sh            # copy results/ back to Mac
```

---

## Phase 2 — Both Jetsons (recommended)

### 1. Config (Mac)

```bash
# cluster/spark_cluster.env
export JETSON2_IP=192.168.1.205
export JETSON2_ENABLED=1
```

### 2. Setup Jetson #2

```bash
ssh-copy-id bvdung@192.168.1.205
ssh bvdung@192.168.1.205 "mkdir -p ~/Thesis_IDS/cluster"
scp cluster/spark_cluster.env bvdung@192.168.1.205:~/Thesis_IDS/cluster/

ssh bvdung@192.168.1.205
cd ~/Thesis_IDS/raspberry && ./scripts/setup_jetson.sh
cd ~/Thesis_IDS && source cluster/load_cluster_env.sh && ./cluster/start_worker.sh
```

Or sync full project from Mac:

```bash
./cluster/sync_workspace.sh
```

### 3. Mac — verify

```bash
source cluster/load_cluster_env.sh
./cluster/check_cluster.sh     # 2 workers ALIVE, parquet on both
```

### 4. Edge SOICT (2 Jetsons)

| Jetson | IP | `.env` template | Role |
|--------|-----|-----------------|------|
| #1 | `192.168.1.50` | `.env.jetson1.example` | `EDGE_NODE_ROLE=anomaly_gate` |
| #2 | `192.168.1.205` | `.env.jetson2.example` | `EDGE_NODE_ROLE=classifier` |

Both point Kafka/DB to `MAC_IP` (`192.168.1.165`). Details: [raspberry/JETSON_DISTRIBUTED.md](../raspberry/JETSON_DISTRIBUTED.md)

---

## Where training output is saved

| What | Location (cluster mode) |
|------|-------------------------|
| Experiment reports, CSV, PNG | `~/Thesis_IDS/results/ml_0X_.../` on **Jetson #1** |
| Shared config | `~/Thesis_IDS/results/shared/best_config.json` |
| Edge PySpark model | `~/Thesis_IDS/raspberry/model/` (after `save_model.py`) |
| Autoencoder gate | `raspberry/model/anomaly_*.pkl` (after `ml_08`) |

**Jetson #2 does not store ML results** — it only runs Spark executors.

Pull to Mac after training:

```bash
./cluster/pull_results.sh
```

Or manually:

```bash
rsync -avz bvdung@192.168.1.50:~/Thesis_IDS/results/ ~/Desktop/Thesis_IDS/results/
```

**Warning:** `sync_workspace.sh` pushes Mac → Jetson with `--delete`. Pull results **before** syncing if Mac has an empty `results/` folder.

---

## Spark tuning (8GB Super Kit, both nodes identical)

| Variable | Value |
|----------|-------|
| `SPARK_WORKER_MEMORY` | `4g` |
| `SPARK_WORKER_CORES` | `4` |
| `SPARK_EXECUTOR_MEMORY` | `2g` |
| `SPARK_DRIVER_MEMORY` | `3g` (Jetson #1 driver) |
| `SPARK_SHUFFLE_PARTITIONS` | `16` |

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
./cluster/pull_results.sh
```

---

## Key environment variables

| Variable | Role |
|----------|------|
| `IDS_SPARK_CLUSTER=1` | Enable cluster mode in `shared_utils.py` |
| `SPARK_MASTER` | `spark://<MAC_IP>:7077` |
| `SPARK_DRIVER_HOST` | Jetson #1 IP (`192.168.1.50`) |
| `JETSON2_ENABLED` | `0` = skip SSH/sync/stop for Jetson #2 |
| `IDS_MAC_ROOT` | Project root on Mac (sync source) |
| `IDS_CLUSTER_DATA_DIR` | Parquet path on Jetson (`.../data`) |
| `SPARK_EXECUTOR_MEMORY` | `2g` (Jetson Super Kit 8GB) |
| `SPARK_WORKER_MEMORY` | `4g` per worker |
| `SPARK_DRIVER_MEMORY` | `3g` (driver on Jetson #1) |

**Mac-only** (local Spark OK): `ml_00`, `save_model.py` with `IDS_ALLOW_LOCAL_SPARK=1`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No route to host` / `:7077` | Wrong `MAC_IP` — run `ipconfig getifaddr en0`, update `spark_cluster.env`, scp to Jetsons, restart master + workers |
| Mac firewall | Allow Java incoming; open ports 7077, 8080 |
| `BASH_SOURCE parameter not set` | Use `source cluster/load_cluster_env.sh` (zsh fix applied) |
| `rsync from spark://...` | Old scripts; master/worker now `unset SPARK_MASTER` |
| SSH `Permission denied` | Correct `JETSON_SSH_USER`, `ssh-copy-id`, enable `PasswordAuthentication yes` on Jetson |
| 0 workers on UI | Run `./cluster/start_worker.sh` **on each Jetson**; verify `MAC_IP` |
| `scp: ... No such file or directory` | `ssh jetson "mkdir -p ~/Thesis_IDS/cluster"` then `./cluster/sync_workspace.sh` |
| `SPARK_HOME=/opt/spark` | `unset SPARK_HOME` before starting worker |
| `Spark sbin not found` | Install pyspark in venv; script uses `pyspark.__path__[0]` |
| `Java gateway exited` | `sudo apt install default-jdk`; verify `java -version` on Jetson |
| `No module named pandas` | `pip install -r cluster/requirements_ml_driver_min.txt` |
| pip `IncompleteRead` on Jetson | Install packages one-by-one; use `_min.txt` first |
| `Missing parquet on Jetson` | `./cluster/sync_workspace.sh`; verify `IDS_MAC_ROOT` on Mac |
| Kafka `NoBrokersAvailable` | Wait 30s after `docker compose up`; use `--bootstrap localhost:9092` on Mac |
| Kafka `Restarting (1)` / `InconsistentClusterIdException` | Stale volume — `docker compose stop kafka && docker volume rm raspberry_kafka_data && docker compose up -d kafka` |
| OOM on Jetson | Reduce `SPARK_EXECUTOR_MEMORY`; increase swap |

---

## Code layout

| File | Role |
|------|------|
| `shared_utils.py` | `create_spark_session()` — cluster + auto JAVA_HOME |
| `cluster/load_cluster_env.sh` | Load config (zsh + bash) |
| `cluster/sync_workspace.sh` | rsync Mac → Jetson(s) |
| `cluster/pull_results.sh` | rsync Jetson #1 → Mac (`results/`, `raspberry/model/`) |
| `cluster/run_ml_remote.sh` | SSH driver on Jetson #1 |
| `cluster/start_master_mac.sh` | Spark master on Mac |
| `cluster/start_worker.sh` | Spark worker on Jetson |
| `cluster/stop_cluster.sh` | Stop master + workers |
| `cluster/check_cluster.sh` | Health check |
