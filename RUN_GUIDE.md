# Execution Guide — IDS Thesis Project

This guide provides step-by-step instructions to reproduce the entire pipeline, from data preparation and training on a PC/Mac to deployment and evaluation on a Raspberry Pi.

---

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
python data_preparation.py
```
*Output: `data/train_data.parquet` and `data/test_data.parquet`*

### 3. Run Experiments (Baseline to SHAP)
Each script evaluates models and generates reports in their respective `exp*_results/` folders.
```bash
python exp0_baseline_full.py            # Baseline (all features)
python exp1_rf_feature_importance.py    # Random Forest Feature Importance (generates importance.csv)
python exp3_pca.py                      # PCA Dimensionality Reduction
python exp5_shap_explainability.py      # SHAP XAI Analysis
python exp6_shap_feature_selection.py   # SHAP Feature Selection (Top-K)
python exp7_comparison.py               # Cross-method + Robustness + Drift + Statistical validity
python exp2_gridsearch_cv.py            # Hyperparameter Tuning on best_config.json from Exp7
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
