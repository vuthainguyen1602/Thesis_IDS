#!/bin/bash
# Run the full IDS pipeline on a SINGLE machine (Mac/PC), in dependency order.
# Fills results/ used by the papers. For the distributed cluster, use
# cluster/reproduce_cluster.sh instead.
#
# Usage:
#   ./run_all_local.sh                # leak-free run (default, for the papers)
#   IDS_KEEP_PORT_FEATURES=1 IDS_DEDUP_ON_FEATURES=0 IDS_DATA_DIR=./data_leaky \
#       ./run_all_local.sh prepare baseline   # leaky run for the ablation table
#
# Prereqs:
#   - Java 11+ (java -version)
#   - pip install pyspark==3.4.1 xgboost shap scipy pyarrow scikit-learn \
#                 pandas numpy matplotlib seaborn joblib
#   - Raw CICIDS2017 CSVs in $IDS_RAW_DATA_DIR (default ./ids-2017)
set -uo pipefail

# Single-machine Spark
export IDS_ALLOW_LOCAL_SPARK=1
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"   # avoids hostname-resolution errors
export IDS_DATA_DIR="${IDS_DATA_DIR:-./data}"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-$(command -v python3)}"

run() { echo -e "\n=========== $1 ==========="; python3 "$1" || { echo "[FAIL] $1"; exit 1; }; }

STEPS="${*:-all}"
do_step() { case "$STEPS" in *"$1"*|all) return 0;; *) return 1;; esac; }

do_step prepare  && run ml_00_prepare_cicids2017.py
do_step baseline && run ml_01_baseline_all_features.py
do_step rf       && run ml_02_feature_selection_rf.py        # -> feature_importance.csv
do_step shap     && run ml_06_feature_selection_shap.py      # -> shap_feature_importance.csv
do_step pca      && run ml_04_dimensionality_reduction_pca.py
do_step cross    && run ml_07_cross_method_comparison.py     # needs rf+shap; writes best_config.json
do_step tune     && run ml_03_hyperparameter_tuning.py       # needs best_config.json
do_step shapxai  && run ml_05_shap_explainability.py         # needs rf importance
do_step multi    && run ml_09_multiclass_eval.py             # per-attack metrics
# Edge anomaly gate: export feature list + a model first, then train the AE gate
if do_step gate; then
  run raspberry/scripts/save_model.py
  run ml_08_anomaly_gate_autoencoder.py                      # -> gate_operating_points.csv
fi

echo -e "\n[OK] Done. Results in: $(pwd)/results/"
echo "Collect figures/CSVs for the papers:  ./thesis/collect_results.sh"
