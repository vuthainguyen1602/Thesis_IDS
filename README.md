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
| **Exp 3** | Dimensionality Reduction using PCA (k=15/25/35) |
| **Exp 5** | SHAP Explainability - XGBoost model interpretation |
| **Exp 6** | Feature Selection using SHAP Importance (Top-20/30/40) |
| **Exp 7** | Cross-Experiment Comparison (Baseline vs RF vs SHAP vs PCA) |

---

## Project Structure

```
Thesis_IDS/
├── README.md                        # This file
├── shared_utils.py                  # Core library (Spark config, models, metrics, plots)
├── data_preparation.py              # CICIDS2017 data preparation script
├── exp0_baseline_full.py            # Experiment 0: Baseline
├── exp1_rf_feature_importance.py    # Experiment 1: RF Feature Selection
├── exp2_gridsearch_cv.py            # Experiment 2: Grid Search + CV
├── exp3_pca.py                      # Experiment 3: PCA
├── exp5_shap_explainability.py      # Experiment 5: SHAP XAI
├── exp6_shap_feature_selection.py   # Experiment 6: SHAP Feature Selection
├── exp7_comparison.py               # Experiment 7: Cross-Experiment Comparison
├── feature_importance.csv           # RF Feature Importance (output from Exp 1)
├── data/                            # Processed data (parquet format)
│   ├── train_data.parquet/
│   └── test_data.parquet/
├── exp0_results/                    # Exp 0 results (charts + HTML report)
├── exp1_results/                    # Exp 1 results
├── exp2_results/                    # Exp 2 results
├── exp3_results/                    # Exp 3 results
├── exp5_results/                    # Exp 5 results
├── exp6_results_shap/               # Exp 6 results
├── exp7_comparison/                 # Exp 7 results (cross-method comparison)
├── latex/                           # LaTeX pseudocode for algorithms
│   ├── algorithm_bagging.tex
│   └── algorithm_voting.tex
├── Thesis_IDS_RoEduNet/             # Experiments for RoEduNet dataset
│   ├── shared_utils.py
│   ├── data_preparation.py
│   ├── exp0_baseline_full.py
│   ├── exp1_rf_feature_importance.py
│   └── exp2_gridsearch_cv.py
└── raspberry/                        # Raspberry Pi Edge IDS Component
    ├── README.md                      # Edge-specific documentation
    ├── docker-compose.yml             # Kafka, Postgres, InfluxDB infra
    ├── edge/                          # Core inference engine (PySpark)
    ├── alerting/                      # Alerting (Email, Slack)
    ├── storage/                       # Storage (Postgres, InfluxDB)
    ├── sender/                        # Traffic simulation sender
    └── scripts/                       # Setup and Benchmark scripts
```

---

## System Requirements

- **Python**: 3.9+
- **Java JDK**: 17 (required for Apache Spark)
- **RAM**: minimum 8GB (16GB recommended)
- **Disk**: ~5GB for CICIDS2017 dataset

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

### 6. Configure JAVA_HOME

Open `shared_utils.py` and update the Java path if needed:

```python
# Lines 27-28 in shared_utils.py
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
```

Common paths:
| OS | Path |
|----|------|
| macOS (Homebrew) | `/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home` |
| macOS (Oracle) | `/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home` |
| Ubuntu | `/usr/lib/jvm/java-17-openjdk-amd64` |

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

Extract all CSV files into a directory (default: `/Users/thainguyenvu/Desktop/ids-2017`).

Required CSV files:
```
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

### Step 2: Update Paths

Open `data_preparation.py` and update `INPUT_PATH`:

```python
INPUT_PATH = "/path/to/your/csv/directory"
```

### Step 3: Run Data Preparation

```bash
python data_preparation.py
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

1. **PC Training**: Data Prep -> exp0 to exp7
2. **Model Export**: Save models for RPi
3. **Infrastructure**: Start Docker services
4. **Edge Deployment**: RPi Setup -> Kafka Consumer
5. **Monitoring**: Grafana Dashboards

---

## Running Experiments

### Recommended Order

```bash
# Step 0: Data preparation (run once)
python data_preparation.py

# Step 1: Baseline - evaluate all features
python exp0_baseline_full.py

# Step 2: Feature Selection with RF Importance
python exp1_rf_feature_importance.py

# Step 3: Hyperparameter Optimization (Grid Search + CV)
python exp2_gridsearch_cv.py

# Step 4: Dimensionality Reduction with PCA
python exp3_pca.py

# Step 5: SHAP Explainability
python exp5_shap_explainability.py

# Step 6: Feature Selection with SHAP
python exp6_shap_feature_selection.py

# Step 7: Cross-Experiment Comparison
python exp7_comparison.py
```

> **Important:** Run `exp1` before `exp2` because exp2 needs `feature_importance.csv` from exp1. Run `exp5` before `exp6` for SHAP vs RF comparison.

### Running on RoEduNet Dataset

```bash
cd Thesis_IDS_RoEduNet/
python data_preparation.py
python exp0_baseline_full.py
python exp1_rf_feature_importance.py
python exp2_gridsearch_cv.py
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
- Evaluation metric: **PR-AUC** (correlated with F1 for binary classification)
- Compares Tuned vs Default model performance

### Experiment 3: PCA Dimensionality Reduction
- Analyzes Explained Variance to determine optimal number of components
- Evaluates all algorithms with **PCA k=15, 25, 35**
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
- Runs **4 methods** side-by-side: Baseline, RF Top-30, SHAP Top-30, PCA k=35
- Generates: Grouped F1 bar chart, F1 heatmap, Best-F1 summary, CSV export
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

## Edge Deployment (Raspberry Pi)

The system is designed for a **Split Deployment** architecture:
- **Cloud/PC**: Runs the training pipeline (PySpark) and central infrastructure (Kafka, PostgreSQL, InfluxDB, Grafana).
- **Edge (RPi 4B)**: Runs the real-time inference engine, preprocessing JSON network flows and making predictions using a pre-trained `PipelineModel`.

### Features
- **Real-time Inference**: < 100ms latency per batch.
- **Performance Monitoring**: Real-time CPU/RAM/Thermal tracking via InfluxDB.
- **Alerting**: Multi-channel alerts (Email via Mailtrap, Slack Incoming Webhooks).
- **Dashboards**: Pre-configured Grafana visualizations for both performance and security metrics.

See the **[raspberry/README.md](raspberry/README.md)** for detailed hardware setup and edge-specific commands.

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

1. **Memory:** Spark defaults to `local[4]` (4 cores). If you encounter OutOfMemory errors, adjust in `shared_utils.py`:
   ```python
   os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[2] pyspark-shell'
   ```

2. **Absolute Paths:** Experiment files use absolute paths. Update them to match your system.

3. **Runtime:** Each experiment takes approximately **30 minutes to 2 hours** depending on hardware.
