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
├── reporting/                       # Modular reporting library [NEW]
│   └── report_generator.py          # HTML/CSS report generation logic
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
├── raspberry/                       # Raspberry Pi Edge IDS Component
└── ...
```

---

## Environment Configuration

The project uses a centralized environment variable system for path management, ensuring portability across different machines without code changes.

| Variable | Description | Default Fallback |
|----------|-------------|------------------|
| `IDS_ROOT` | Project root directory | Script directory (`__file__`) |
| `IDS_RAW_DATA_DIR` | Location of raw CICIDS2017 CSVs | `IDS_ROOT/ids-2017` |
| `IDS_DATA_DIR` | Location to save/load Parquet | `IDS_ROOT/data` |

### Setting variables (Example)
```bash
export IDS_ROOT="/Users/name/Desktop/Thesis_IDS"
export IDS_RAW_DATA_DIR="/Volumes/ExternalSSD/Dataset/ids-2017"
```

---

## Project Standards: Scientific Sanitization

This codebase has been refactored to meet **higher scientific standards** for thesis submission:

1.  **Logic-First Architecture**: All redundant comments and legacy docstrings have been removed to ensure the reviewer focuses purely on the technical implementation.
2.  **Modular Reporting**: HTML/CSS visualization logic has been extracted into the `reporting/` module to keep the core experiment code concise.
3.  **Portability**: Hardcoded absolute paths have been eliminated in favor of the `IDS_ROOT` dynamic discovery pattern.

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

Extract all CSV files into a directory.

### Step 2: Configure Paths
Setting the environment variables is the recommended way to configure paths:
```bash
export IDS_RAW_DATA_DIR="/path/to/your/csv/directory"
```
Or you can let the script default to searching for a folder named `ids-2017` inside your project root.

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

# Step 3: Dimensionality Reduction with PCA
python exp3_pca.py

# Step 4: SHAP Explainability
python exp5_shap_explainability.py

# Step 5: Feature Selection with SHAP
python exp6_shap_feature_selection.py

# Step 6: Cross-Experiment + Robustness + Drift + Statistical tracks
python exp7_comparison.py

# Step 7: Hyperparameter Optimization on best config from Exp7
python exp2_gridsearch_cv.py
```

> **Important:** Run `exp7` before `exp2` because `exp2` reads `best_config.json` from `exp7`.  
> Optional robustness dataset: set `IDS_ROBUST_DATA_DIR` containing `test_data.parquet` before running `exp7`.

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
