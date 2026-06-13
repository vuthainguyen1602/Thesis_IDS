# Execution Guide — IDS Thesis Project

This guide covers **distributed cluster mode** (Mac + 2× Jetson, recommended) and legacy local/RPi paths.

**Cluster guide:** [cluster/DISTRIBUTED_CLUSTER.md](cluster/DISTRIBUTED_CLUSTER.md)

---

## Distributed cluster (Mac + 2× Jetson Super Kit 8GB)

| Node | IP (lab) | Role |
|------|----------|------|
| Mac | `192.168.1.165` | Spark Master, Docker, `ml_00`, sync/pull |
| Jetson #1 | `192.168.1.50` | Worker + ML driver, `results/` |
| Jetson #2 | `192.168.1.205` | Worker, edge classifier |

### 1. Configure cluster

```bash
cp cluster/spark_cluster.env.example cluster/spark_cluster.env
# Edit MAC_IP (ipconfig getifaddr en0), JETSON IPs, JETSON_SSH_USER, IDS_MAC_ROOT
source cluster/load_cluster_env.sh
```

### 2. Start cluster

```bash
# Mac
./cluster/start_master_mac.sh
cd raspberry && docker compose up -d

# Each Jetson (SSH)
cd ~/Thesis_IDS && source cluster/load_cluster_env.sh && ./cluster/start_worker.sh

# Mac — verify: http://192.168.1.165:8080 → 2 workers ALIVE
./cluster/check_cluster.sh
```

### 3. Preprocess + sync + train

```bash
python ml_00_prepare_cicids2017.py          # Mac only
./cluster/sync_workspace.sh
./cluster/run_ml_remote.sh ml_01_baseline_all_features.py
./cluster/pull_results.sh                   # Jetson #1 → Mac
```

Full pipeline:

```bash
./cluster/reproduce_cluster.sh fair    # or thesis / soict
./cluster/pull_results.sh
./papers/fair2026/collect_results.sh
```

---

## Local mode (Mac only — debug)

### 1. Configure the Environment
Set the project root and data directory via environment variables to ensure portability.
```bash
export IDS_ROOT="$(pwd)"
export IDS_RAW_DATA_DIR="/path/to/your/ids-2017/csvs"
# Optional: additional holdout dataset for Exp7 robustness track
# export IDS_ROBUST_DATA_DIR="/path/to/robustness_parquet_dir"
```

### 2. Preprocess the Dataset
Prepare the CICIDS2017 dataset for Spark.
```bash
python ml_00_prepare_cicids2017.py
```
*Output: `data/train_data.parquet` and `data/test_data.parquet`*

### 3. Run Experiments (Baseline to SHAP)
Each script evaluates models and generates reports under `results/<experiment_name>/` (local) or on Jetson #1 driver (cluster).
```bash
python ml_01_baseline_all_features.py            # Baseline (all features)
python ml_02_feature_selection_rf.py           # Random Forest Feature Importance (generates importance.csv)
python ml_04_dimensionality_reduction_pca.py   # PCA Dimensionality Reduction
python ml_05_shap_explainability.py            # SHAP XAI Analysis
python ml_06_feature_selection_shap.py         # SHAP Feature Selection (Top-K)
python ml_07_cross_method_comparison.py        # Cross-method + Robustness + Drift + Statistical validity
python ml_03_hyperparameter_tuning.py          # Hyperparameter Tuning on best_config.json from ml_07
```

---

## Stage 2: Model Export for Edge (PC/Mac)

### 4. Save PySpark Pipeline for RPi
Export the best-performing models (Decision Tree/RF/GBT) as PipelineModels for the Edge engine.
```bash
python raspberry/scripts/save_model.py      # Save single optimal model (DT)
python raspberry/scripts/save_all_models.py  # Save multiple models for benchmarking
```
*Output: `raspberry/model/ids_pipeline_model/`*

---

## Stage 3: Infrastructure Setup (PC/Mac - Docker)

### 5. Start Centralized Services
Kafka, PostgreSQL, and InfluxDB run on the Mac/PC to store results and relay traffic.
```bash
cd raspberry/
docker compose up -d
```
*Check status: `docker compose ps`*

---

## Stage 4: Edge Deployment (Raspberry Pi)

### 6. Remote Setup
Connect to the Raspberry Pi and install the environment.
```bash
ssh pi@<rpi-ip>
cd ~/raspberry
chmod +x scripts/setup_raspberry.sh
./scripts/setup_raspberry.sh
```

### 7. Copy Model to RPi
From your **PC/Mac**, send the exported model to the Pi.
```bash
scp -r ~/Thesis_IDS/raspberry/model/* pi@<rpi-ip>:~/raspberry/model/
```

### 8. Start the IDS Consumer
On the **Raspberry Pi**, start the real-time inference engine.
```bash
cd ~/raspberry
source venv/bin/activate
python edge/kafka_consumer.py
```

---

## Stage 5: Evaluation & Monitoring

### 9. Simulate Network Traffic (PC/Mac)
Stream CSV data rows to the RPi via Kafka.
```bash
cd raspberry/
python sender/data_sender.py --rate 100
```

### 10. Benchmark Performance (Raspberry Pi)
Measure throughput and latency on the edge device.
```bash
python scripts/benchmark.py --samples 500
python scripts/benchmark_all.py  # Multi-model comparison
```

### 11. Dashboard Monitoring (Browser)
Open Grafana on your **PC/Mac** to view live metrics.
- **URL**: `http://localhost:3000` (User: `admin` / `admin`)
- **Action**: Import JSON from `raspberry/dashboard/grafana_dashboard.json`
