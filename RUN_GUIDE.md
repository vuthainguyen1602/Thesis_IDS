# Execution Guide — IDS Thesis Project

Step-by-step for **distributed cluster mode** (Mac + 2× Jetson Orin Nano Super 8GB).

**Detailed guide:** [cluster/DISTRIBUTED_CLUSTER.md](cluster/DISTRIBUTED_CLUSTER.md)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Same LAN | Mac + Jetsons on one WiFi (e.g. `192.168.1.x`), mutual `ping` |
| `cluster/spark_cluster.env` | Copy from `.example`; set `MAC_IP`, Jetson IPs, `JETSON_SSH_USER`, `IDS_MAC_ROOT` |
| SSH keys | `ssh-copy-id <user>@<JETSON1_IP>` (and #2) |
| Parquet data | `python ml_00_prepare_cicids2017.py` on Mac (once) |

### Lab IPs (reference)

| Node | IP | Role |
|------|-----|------|
| Mac | `192.168.1.165` | Spark Master, Docker, sync/pull |
| Jetson #1 | `192.168.1.50` | Worker + ML driver, `results/` |
| Jetson #2 | `192.168.1.205` | Worker, edge classifier |

---

## One-time setup

### 1. Configure cluster (Mac)

```bash
cd ~/Desktop/Thesis_IDS
cp cluster/spark_cluster.env.example cluster/spark_cluster.env
# Edit MAC_IP, JETSON1_IP, JETSON2_IP, JETSON_SSH_USER, IDS_MAC_ROOT
source cluster/load_cluster_env.sh
```

Also set Kafka in `jetson/docker-compose.yml`:

```yaml
KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka:29092,EXTERNAL://<MAC_IP>:9092
```

### 2. Setup Jetsons (SSH, once per device)

```bash
ssh <user>@<JETSON_IP>
cd ~/Thesis_IDS/jetson && ./scripts/setup_jetson.sh
exit
```

### 3. Data + Docker (Mac, once)

```bash
python ml_00_prepare_cicids2017.py
cd jetson && docker compose up -d
python scripts/init_kafka_topics.py --partitions 2 --bootstrap localhost:9092
```

---

## Daily run — every training session

### Step 1 — Mac: stop old cluster, start master

```bash
cd ~/Desktop/Thesis_IDS
source cluster/load_cluster_env.sh

# Confirm Mac IP still matches spark_cluster.env
ipconfig getifaddr en0

./cluster/stop_cluster.sh
./cluster/start_master_mac.sh
sleep 3
```

### Step 2 — Jetson #1: start worker

```bash
ssh <user>@192.168.1.50

export SPARK_HOME=$(python3 -c "import pyspark; print(pyspark.__path__[0])")
unset SPARK_MASTER
$SPARK_HOME/sbin/stop-worker.sh 2>/dev/null || true
pkill -f SparkSubmit 2>/dev/null || true

cd ~/Thesis_IDS
unset SPARK_HOME
source cluster/load_cluster_env.sh
./cluster/start_worker.sh
exit
```

Expected output:

```
[INFO] Spark worker (...) -> spark://192.168.1.165:7077 (4 cores, 5g)
[OK] Worker started
```

### Step 3 — Jetson #2: start worker

Same commands as Step 2, but SSH to `192.168.1.205`.

### Step 4 — Mac: verify cluster

```bash
./cluster/check_cluster.sh
```

Open http://192.168.1.165:8080 — expect:

- **Workers:** 2 ALIVE (4 cores, 5 GiB each)
- **Running Applications:** 0 (no stale 7h zombie apps)

Connectivity check from Jetson:

```bash
ssh <user>@192.168.1.50 "ping -c 2 192.168.1.165 && nc -zv 192.168.1.165 7077"
```

### Step 5 — Mac: run everything (one command)

Once the cluster is up (Steps 1–4), a single command runs the full pipeline
(offline Spark + edge benchmarks) and produces the data for **thesis + FAIR + SOICT**:

```bash
./run_all.sh
```

It auto-starts the Mac master, runs `ml_00`→`ml_11` in dependency order, then the
edge benchmarks. Progress/timing land in `output/run_all_logs/`; if interrupted,
just re-run `./run_all.sh` — completed steps are skipped (resume markers in
`.run_all_state/`).

| Command | Effect |
|---------|--------|
| `./run_all.sh offline` \| `edge` | run one phase only |
| `STOP_ON_ERROR=0 ./run_all.sh` | keep going on failure, report at end |
| `FORCE=1 ./run_all.sh` | re-run from scratch (ignore resume markers) |
| `RUN_STREAMING=1 ./run_all.sh edge` | include end-to-end streaming benchmark (needs Docker+Kafka) |

To debug a single step instead:

```bash
./cluster/sync_workspace.sh          # if code/config changed on Mac
./cluster/run_ml_remote.sh ml_01_baseline_all_features.py
```

On first run, Jetson driver auto-installs: `xgboost`, `shap`, `scipy`. Log should show:

```
[OK] Core ML deps ready
[INFO] XGBoost backend available
```

(LightGBM was removed — x86_64-only native libs, not runnable on the ARM64 Jetson.)

New Spark app on Master UI should show **Cores > 0** (typically 8), not `WAITING` with 0 cores.

### Step 6 — Mac: pull results

```bash
./cluster/pull_results.sh
```

Results live on Jetson #1 at `~/Thesis_IDS/results/` until pulled.

---

## Full pipeline (papers / thesis)

```bash
./cluster/reproduce_cluster.sh fair    # or thesis / soict
./cluster/pull_results.sh
./papers/fair2026/collect_results.sh   # if applicable
```

### ML script order (cluster)

| Order | Script | Notes |
|-------|--------|-------|
| 0 | `ml_00_prepare_cicids2017.py` | Mac only |
| 1 | `ml_01_baseline_all_features.py` | 8 models (incl. XGBoost if deps OK) |
| 2 | `ml_02_feature_selection_rf.py` | |
| 3 | `ml_04_dimensionality_reduction_pca.py` | |
| 4 | `ml_05_shap_explainability.py` | Needs xgboost + shap |
| 5 | `ml_06_feature_selection_shap.py` | |
| 6 | `ml_07_cross_method_comparison.py` | Run before ml_03 |
| 7 | `ml_03_hyperparameter_tuning.py` | Reads `best_config.json` from ml_07 |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App `WAITING`, **0 cores** | Kill apps on Spark UI; `pkill -f SparkSubmit` on Jetson #1; restart workers |
| `(0 + 0) / 1` stuck > 15 min | Check workers ALIVE; verify executors registered on Master UI |
| `No route to host` | Mac on wrong network or stale `MAC_IP` — update config, sync, restart |
| No XGBoost in results | Re-run `run_ml_remote.sh`; or on Jetson: `pip install -r cluster/requirements_ml_driver.txt` |
| MLP slow | Normal; tree models (RF, GBT, XGBoost) finish much faster |
| Mac changed IP | Update `MAC_IP`, `docker-compose.yml` Kafka listener, `sync_workspace.sh`, restart all |

Kill stale apps from Mac:

```bash
curl "http://192.168.1.165:8080/app/kill/?id=<APP_ID>&terminate=true"
```

---

## Edge deployment (SOICT)

After ML + `save_model.py`, see [jetson/JETSON_DISTRIBUTED.md](jetson/JETSON_DISTRIBUTED.md).
