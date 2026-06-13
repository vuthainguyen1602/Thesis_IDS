# Intrusion Detection System (IDS) - Thesis Project

Network Intrusion Detection System using **Machine Learning** on **Apache Spark (PySpark)**, evaluated on the **CICIDS2017** and **RoEduNet** datasets.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running Experiments](#running-experiments)
- [Experiment Descriptions](#experiment-descriptions)
- [Output](#output)
- [Edge Deployment (Raspberry Pi)](#edge-deployment-raspberry-pi)
- [Execution Guide](#execution-guide)
- [Code Architecture](#code-architecture)
- [Notes](#notes)

---

## Overview

This project evaluates **9 classification algorithms** combined with **3 Ensemble Learning methods** for binary network intrusion detection (Attack vs. Benign).

### Algorithms

| Type | Algorithms |
|------|-----------|
| **Standalone** | Decision Tree, Logistic Regression, SVM, Naive Bayes, Random Forest, GBT, XGBoost, LightGBM, MLP |
| **Ensemble** | Hybrid Bagging (Top-3, 3-2-2 Weighted), Majority Voting (Top-3 by F1), Stacking |

### Experiments

| Exp | Description |
|-----|-------------|
| **Exp 0** | Baseline - All features (no dimensionality reduction) |
| **Exp 1** | Feature Selection using RF Feature Importance (Top-20/30/40) |
| **Exp 2** | Hyperparameter Optimization with Grid Search + Cross-Validation |
| **Exp 3** | Dimensionality Reduction using PCA (k=20/30/40) |
| **Exp 5** | SHAP Explainability - XGBoost model interpretation |
| **Exp 6** | Feature Selection using SHAP Importance (Top-20/30/40) |
| **Exp 7** | Cross-Experiment + Robustness + Drift + Statistical Comparison |

---

## Project Structure

```
Thesis_IDS/
├── README.md                        # This file
├── shared_utils.py                  # Core library (Spark config, models, metrics, plots)
├── reporting/                       # Modular reporting library
│   └── report_generator.py          # HTML/CSS report generation logic
├── cluster/                         # Spark cluster Mac + 2× Jetson (DISTRIBUTED_CLUSTER.md)
├── ml_00_prepare_cicids2017.py      # CICIDS2017 data preparation
├── ml_01_baseline_all_features.py … ml_08_*.py  # ML experiments
├── data/                            # Processed data (parquet format)
├── raspberry/                       # Edge IDS (RPi / Jetson Nano)
│
├── thesis/                          # Deliverable #1 — thesis (manuscript + reproduce)
├── papers/
│   ├── fair2026/                    # Deliverable #2 — FAIR'2026 paper
│   └── soict2026/                   # Deliverable #3 — SOICT 2026 paper
└── ...
```

### Three deliverables (one codebase)

| Folder | Deliverable | Reproduce |
|--------|-------------|-----------|
| `thesis/` | Full thesis | `./thesis/reproduce.sh` |
| `papers/fair2026/` | ML paper / FAIR'2026 | `./papers/fair2026/reproduce.sh` |
| `papers/soict2026/` | Edge paper / SOICT 2026 | `./papers/soict2026/reproduce.sh` |

Shared execution code lives at the repo root (`ml_*.py`, `raspberry/`). Each folder above holds manuscript, figures, tables, and result-collection scripts.

### ML scripts (run order)

| Script | Description |
|--------|-------------|
| `ml_00_prepare_cicids2017.py` | CSV → parquet |
| `ml_01_baseline_all_features.py` | Baseline |
| `ml_02_feature_selection_rf.py` | RF Top-K |
| `ml_03_hyperparameter_tuning.py` | Grid search |
| `ml_04_dimensionality_reduction_pca.py` | PCA |
| `ml_05_shap_explainability.py` | SHAP XAI |
| `ml_06_feature_selection_shap.py` | SHAP Top-K |
| `ml_07_cross_method_comparison.py` | Cross-method + drift |
| `ml_08_anomaly_gate_autoencoder.py` | Anomaly gate (edge) |

### Distributed cluster (Mac + 2× Jetson Nano) — **required**

All `./papers/*/reproduce.sh` and `./thesis/reproduce.sh` run **only in distributed mode** via Spark cluster:

```
Mac (192.168.1.x)     → Spark Master :7077, Docker (Kafka/Postgres/Grafana)
Jetson #1             → Spark Worker + PySpark driver (ML scripts)
Jetson #2 (optional)  → Second Spark Worker (+ edge classifier for SOICT)
```

**Full guide:** [cluster/DISTRIBUTED_CLUSTER.md](cluster/DISTRIBUTED_CLUSTER.md)

#### Phase 1 — Try 1 Jetson first (recommended)

```bash
cp cluster/spark_cluster.env.example cluster/spark_cluster.env
# Edit: MAC_IP, JETSON1_IP, JETSON_SSH_USER, IDS_MAC_ROOT, IDS_RAW_DATA_DIR
export JETSON2_ENABLED=0          # in spark_cluster.env — skip Jetson #2

# Mac — one-time setup
source cluster/load_cluster_env.sh
./cluster/start_master_mac.sh
cd raspberry && docker compose up -d
python scripts/init_kafka_topics.py --partitions 2 --bootstrap localhost:9092

# Jetson #1 — SSH, one-time setup
ssh-copy-id <user>@<JETSON1_IP>
cd ~/Thesis_IDS/raspberry && ./scripts/setup_jetson.sh
cd ~/Thesis_IDS && source cluster/load_cluster_env.sh && ./cluster/start_worker.sh

# Mac — sync + run ML
./cluster/sync_workspace.sh
./cluster/run_ml_remote.sh ml_01_baseline_all_features.py
```

Verify: http://`<MAC_IP>`:8080 → **1 worker ALIVE**

#### Phase 2 — Add Jetson #2

1. Get Jetson #2 IP: `hostname -I` on the second device  
2. Edit `cluster/spark_cluster.env`:

```bash
export JETSON2_IP=192.168.1.XXX
export JETSON2_ENABLED=1
```

3. SSH + setup (same as Jetson #1):

```bash
ssh-copy-id <user>@<JETSON2_IP>
rsync -avz ~/Desktop/Thesis_IDS/ <user>@<JETSON2_IP>:~/Thesis_IDS/
scp cluster/spark_cluster.env <user>@<JETSON2_IP>:~/Thesis_IDS/cluster/
# On Jetson #2:
cd ~/Thesis_IDS/raspberry && ./scripts/setup_jetson.sh
cd ~/Thesis_IDS && source cluster/load_cluster_env.sh && ./cluster/start_worker.sh
```

4. Mac — sync both + verify:

```bash
source cluster/load_cluster_env.sh
./cluster/sync_workspace.sh      # rsync → Jetson #1 and #2
./cluster/check_cluster.sh       # 2 workers ALIVE, parquet OK
```

5. Edge SOICT (pipeline split): Jetson1 `anomaly_gate`, Jetson2 `classifier` — see [raspberry/JETSON_DISTRIBUTED.md](raspberry/JETSON_DISTRIBUTED.md)

#### Reproduce papers / thesis

```bash
./cluster/reproduce_cluster.sh fair    # FAIR'2026 ML track
./cluster/reproduce_cluster.sh soict   # SOICT edge track
./cluster/reproduce_cluster.sh thesis  # Full thesis
```

**Mac-only** exceptions: `ml_00_prepare_cicids2017.py`, `save_model.py` (`IDS_ALLOW_LOCAL_SPARK=1`).

#### Cluster environment variables (summary)

| Variable | Mac | Jetson |
|----------|-----|--------|
| `MAC_IP` | Mac WiFi IP (`ipconfig getifaddr en0`) | — |
| `JETSON1_IP` / `JETSON2_IP` | in `spark_cluster.env` | device IP |
| `JETSON_SSH_USER` | SSH user (e.g. `bvdung`) | — |
| `JETSON2_ENABLED` | `0` = 1 Jetson, `1` = 2 Jetsons | — |
| `IDS_MAC_ROOT` | project path on Mac | — |
| `CLUSTER_DRIVER_IDS_ROOT` | — | `/home/<user>/Thesis_IDS` |
| `SPARK_MASTER` | `spark://<MAC_IP>:7077` | same value |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` (init/sender) | `<MAC_IP>:9092` |

---

## Environment Configuration

The project uses a centralized environment variable system for path management, ensuring portability across different machines without code changes.

| Variable | Description | Default Fallback |
|----------|-------------|------------------|
| `IDS_MAC_ROOT` | Project root **on Mac** (for sync) | Script directory |
| `IDS_ROOT` | Project root on **Jetson driver** | Script directory |
| `IDS_RAW_DATA_DIR` | Location of raw CICIDS2017 CSVs | `IDS_ROOT/ids-2017` |
| `IDS_DATA_DIR` | Location to save/load Parquet **on Mac** | `IDS_MAC_ROOT/data` |
| `IDS_CLUSTER_DATA_DIR` | Parquet path on **Jetsons** | `/home/<user>/Thesis_IDS/data` |

### Setting variables (Example — Mac)
```bash
# In cluster/spark_cluster.env (copy from cluster/spark_cluster.env.example)
export IDS_MAC_ROOT="/Users/name/Desktop/Thesis_IDS"
export IDS_RAW_DATA_DIR="/Users/name/Desktop/Thesis_IDS/ids-2017"
export IDS_DATA_DIR="/Users/name/Desktop/Thesis_IDS/data"
```

---

## Project Standards: Scientific Sanitization

This codebase has been refactored to meet **higher scientific standards** for thesis submission:

1.  **Logic-First Architecture**: All redundant comments and legacy docstrings have been removed to ensure the reviewer focuses purely on the technical implementation.
2.  **Modular Reporting**: HTML/CSS visualization logic has been extracted into the `reporting/` module to keep the core experiment code concise.
3.  **Portability**: Hardcoded absolute paths have been eliminated in favor of the `IDS_ROOT` dynamic discovery pattern.

---

## System Requirements

### Mac (Spark master + Docker)
- **Python**: 3.9+
- **Java JDK**: 17 (`brew install openjdk@17`)
- **Docker Desktop**: Kafka, PostgreSQL, InfluxDB, Grafana
- **RAM**: 8GB+ (16GB recommended for data prep)

### Jetson Nano (Spark worker + ML driver / edge)
- **JetPack** / Ubuntu 18.04+ (aarch64)
- **Java**: `default-jdk` (Java 11 OK — auto-detected by `shared_utils.py`)
- **Swap**: 4GB (`setup_jetson.sh` creates it automatically)
- **SSH**: passwordless from Mac (`ssh-copy-id user@jetson-ip`)
- **Disk**: ~5GB for CICIDS2017 parquet (synced from Mac)

### Cluster sizing

| Setup | ML training | Edge SOICT |
|----------|-------------|------------|
| Mac + 1 Jetson | ✅ Trial / debug (`JETSON2_ENABLED=0`) | ✅ `full` mode on one Jetson |
| Mac + 2 Jetsons | ✅ Recommended (2 executors) | ✅ Split `anomaly_gate` + `classifier` |

---

## Installation

### 1. Install Java JDK 17

```bash
# macOS (Homebrew)
brew install openjdk@17

# Ubuntu/Debian
sudo apt install openjdk-17-jdk

# Or download from: https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html
```

Verify installation:

```bash
java -version
# Expected: openjdk version "17.x.x"
```

### 2. Install Core Python Dependencies

```bash
pip install pyspark numpy pandas matplotlib seaborn scikit-learn
```

### 3. Install XGBoost (for Spark)

XGBoost is used as `xgboost.spark.SparkXGBClassifier` — the native Spark-compatible API.

```bash
pip install xgboost
```

> **Requirements:** XGBoost >= 1.7.0 is needed for the `xgboost.spark` module. Verify with:
> ```bash
> python -c "from xgboost.spark import SparkXGBClassifier; print('XGBoost Spark OK')"
> ```

### 4. Install LightGBM (via SynapseML)

LightGBM runs on Spark through **SynapseML** (formerly MMLSpark), which provides a Java/Scala backend.

```bash
pip install synapseml
```

SynapseML requires additional Spark packages at runtime. Add the following to your Spark configuration (already configured in `shared_utils.py`):

```python
# In shared_utils.py → create_spark_session()
.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.4")
.config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
```

> **Verify installation:**
> ```bash
> python -c "from synapse.ml.lightgbm import LightGBMClassifier; print('LightGBM Spark OK')"
> ```
>
> **Troubleshooting:** If you get a Java error, ensure:
> 1. `JAVA_HOME` is set correctly
> 2. Spark can download the SynapseML JAR (requires internet on first run)
> 3. If behind a proxy, pre-download the JAR and use `spark.jars` instead of `spark.jars.packages`

### 5. Install SHAP (for Experiments 5 & 6)

```bash
pip install shap
```

### 6. JAVA_HOME

`shared_utils.py` auto-detects `JAVA_HOME` on Mac and Jetson. Install Java only:

| Host | Command |
|-----|---------|
| Mac | `brew install openjdk@17` |
| Jetson | `sudo apt install -y default-jdk` |

Verify: `java -version` and `echo $JAVA_HOME`

### Quick Verification

Run this to verify all dependencies:

```bash
python -c "
import pyspark; print(f'PySpark: {pyspark.__version__}')
import xgboost; print(f'XGBoost: {xgboost.__version__}')
import shap; print(f'SHAP: {shap.__version__}')
from xgboost.spark import SparkXGBClassifier; print('  → XGBoost Spark: OK')
try:
    from synapse.ml.lightgbm import LightGBMClassifier; print('  → LightGBM Spark: OK')
except: print('  → LightGBM Spark: Not available (optional)')
print('All dependencies OK!')
"
```

---

## Data Preparation

### Step 1: Download CICIDS2017 Dataset

Download from: https://www.unb.ca/cic/datasets/ids-2017.html

Extract all CSV files into a directory.

### Step 2: Configure Paths
Setting the environment variables is the recommended way to configure paths:
```bash
export IDS_RAW_DATA_DIR="/path/to/your/csv/directory"
```
Or you can let the script default to searching for a folder named `ids-2017` inside your project root.

### Step 3: Run Data Preparation

```bash
python ml_00_prepare_cicids2017.py
```

This script will:
1. Merge 8 CSV files into a single DataFrame
2. Handle infinity values, NaN, and duplicates
3. Create binary labels (`label_binary`: 0 = Benign, 1 = Attack)
4. Split into train/test sets (80/20)
5. Save as parquet at `data/train_data.parquet` and `data/test_data.parquet`

>  **Note:** Run this **ONCE**. All experiments read directly from the parquet files.

---

## Execution Guide

For common execution workflows, following the steps in order:

👉 **[RUN_GUIDE.md](RUN_GUIDE.md)**

1. **PC Training**: Data prep → ml_01 … ml_07
2. **Model Export**: Save models for RPi
3. **Infrastructure**: Start Docker services
4. **Edge Deployment**: RPi Setup -> Kafka Consumer
5. **Monitoring**: Grafana Dashboards

---

## Running Experiments

### Recommended Order

```bash
# Step 0: Data preparation (run once)
python ml_00_prepare_cicids2017.py

# Step 1: Baseline - evaluate all features
python ml_01_baseline_all_features.py

# Step 2: Feature Selection with RF Importance
python ml_02_feature_selection_rf.py

# Step 3: Dimensionality Reduction with PCA
python ml_04_dimensionality_reduction_pca.py

# Step 4: SHAP Explainability
python ml_05_shap_explainability.py

# Step 5: Feature Selection with SHAP
python ml_06_feature_selection_shap.py

# Step 6: Cross-Experiment + Robustness + Drift + Statistical tracks
python ml_07_cross_method_comparison.py

# Step 7: Hyperparameter Optimization on best config from Exp7
python ml_03_hyperparameter_tuning.py
```

> **Important:** Run `ml_07` before `ml_03` because `ml_03` reads `best_config.json` from `ml_07`.  
> Optional robustness dataset: set `IDS_ROBUST_DATA_DIR` containing `test_data.parquet` before running `ml_07`.

### Running on RoEduNet Dataset

```bash
cd Thesis_IDS_RoEduNet/
python ml_00_prepare_cicids2017.py
python ml_01_baseline_all_features.py
python ml_02_feature_selection_rf.py
python ml_03_hyperparameter_tuning.py
```

---

## Experiment Descriptions

### Experiment 0: Baseline (All Features)
- Evaluates 9 algorithms + Hybrid Bagging + Majority Voting on **all features**
- Establishes performance baseline for comparison with dimensionality reduction methods

### Experiment 1: RF Feature Importance
- Trains Random Forest to extract **Feature Importance** rankings
- Evaluates all algorithms with **Top-20, Top-30, Top-40** most important features
- Exports `feature_importance.csv` for use in other experiments

### Experiment 2: Grid Search + Cross-Validation
- Optimizes hyperparameters for **RF, GBT, Decision Tree, Logistic Regression** using Grid Search + 3-Fold CV
- **Default: fast mode** (~15% CV sample, DT+RF+LR) — ~20–45 min
- Full tuning: `IDS_EXP2_FULL=1 python ml_03_hyperparameter_tuning.py`
- Extended (XGB/LGBM/MLP): `IDS_EXP2_FULL=1 IDS_EXP2_EXTENDED=1 python ml_03_hyperparameter_tuning.py`
- Evaluation metric: **PR-AUC** (correlated with F1 for binary classification)
- Compares Tuned vs Default model performance

### Experiment 3: PCA Dimensionality Reduction
- Analyzes Explained Variance to determine optimal number of components
- Evaluates all algorithms with **PCA k=20, 30, 40**
- Compares PCA with Feature Selection approach (Exp 1)

### Experiment 5: SHAP Explainability (XAI)
- Explains **XGBoost** predictions using SHAP (SHapley Additive exPlanations)
- Generates: Summary Plot, Bar Plot, Waterfall Plots
- Compares SHAP Importance vs RF Feature Importance

### Experiment 6: SHAP Feature Selection
- Uses SHAP Importance for feature selection instead of RF Importance
- Evaluates all algorithms with **SHAP Top-20, 30, 40** features
- Compares effectiveness: SHAP vs RF Feature Selection

### Experiment 7: Cross-Experiment Comparison
- Runs **4 methods** side-by-side: Baseline, RF Top-30, SHAP Top-30, PCA k=40
- Adds 3 advanced tracks:
  - Robustness holdout evaluation (external test split via `IDS_ROBUST_DATA_DIR` or fallback split)
  - Drift simulation (`Early -> Mid -> Late` and retrain recovery)
  - Multi-seed stability + permutation significance test (top methods)
- Generates: Grouped F1 bar chart, F1 heatmap, Best-F1 summary, robustness/drift/statistical CSV exports
- Comprehensive HTML report comparing all dimensionality reduction approaches

---

## Output

Each experiment generates:

| Output | Description |
|--------|-------------|
| `comparison.png` | Accuracy, Precision, Recall, F1 comparison |
| `train_time.png` | Training time comparison |
| `pred_time.png` | Prediction time comparison |
| `model_size.png` | Model size comparison (MB) |
| `confusion_matrices.png` | Confusion matrices |
| `roc_curves.png` | ROC curves |
| `exp*_report.html` | Comprehensive HTML report (open in browser) |

---

## Edge Deployment (Jetson / Raspberry Pi)

**Split deployment:**
- **Mac**: Spark master, Docker (Kafka, PostgreSQL, InfluxDB, Grafana), `ml_00`, `save_model.py`
- **Jetson #1**: Spark worker, PySpark ML driver, edge pipeline (`full` or `anomaly_gate`)
- **Jetson #2** (optional): Spark worker, edge `classifier` (SOICT split mode)

Jetson **does not need Docker** — set `.env` `KAFKA_BOOTSTRAP_SERVERS`, `POSTGRES_HOST`, `INFLUXDB_URL` to **MAC_IP**.

See:
- **[cluster/DISTRIBUTED_CLUSTER.md](cluster/DISTRIBUTED_CLUSTER.md)** — Spark cluster + distributed ML
- **[raspberry/JETSON_DISTRIBUTED.md](raspberry/JETSON_DISTRIBUTED.md)** — 2-Jetson edge modes
- **[raspberry/README.md](raspberry/README.md)** — Docker, Kafka, Grafana

---

## Code Architecture

### `shared_utils.py` - Core Library

| Module | Description |
|--------|-------------|
| **Spark Configuration** | SparkSession initialization, JVM configuration |
| **Data Processing** | Load parquet, clean data, feature engineering |
| **Classifiers** | 9 ML algorithms with optimized hyperparameters |
| **Ensemble Learning** | Hybrid Bagging (3-2-2), Majority Voting (Top-3 F1) |
| **Evaluation** | Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR |
| **Visualization** | Charts, Confusion Matrices, ROC Curves |
| **Reporting** | HTML report export |
| **SHAP** | SHAP Explainability for XGBoost |

### `reporting/` - Visualization Module [NEW]
- **Modular Design**: Separates complex HTML/CSS templates from experiment logic.
- **Dynamic Reports**: Generates responsive multi-section reports with embedded performance metrics and SHAP visualizations.

### Ensemble Algorithms

**Hybrid Bagging:**
1. Train K base models
2. Select Top-3 by F1-Score
3. Create 3-2-2 ensemble (3 replicas of Rank 1, 2 of Rank 2, 2 of Rank 3)
4. Soft Voting with F1-weighted probabilities

**Majority Voting:**
1. Train K base models
2. Select Top-3 by F1-Score
3. Collect predictions from 3 models
4. Hard Voting: `prediction = 1 if sum > K/2, else 0`

---

## Notes

1. **Memory:** Spark defaults to `local[*]` (all cores). If you encounter OutOfMemory errors, adjust in `shared_utils.py`.

2. **Portable Paths:** All experiment files are now portable. You do **not** need to modify the code to change paths; simply use the `IDS_ROOT` environment variable.

3. **Runtime:** Each experiment takes approximately **30 minutes to 2 hours** depending on hardware.
