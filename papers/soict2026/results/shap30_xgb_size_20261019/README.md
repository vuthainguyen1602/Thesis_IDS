# SHAP Top-30 + XGBoost: model size (SOICT Table 2)

The ml_07 run saved this model from the training cluster, where the distributed save kept only
driver-side metadata (0.003 MB in `results/ml_07_cross_method_comparison/cross_method_summary.csv`).
`measure_size.py` retrains the same configuration (ml_07 SHAP Top-30 features, `get_classifiers`
XGBoost parameters, train/val split seed 42) on one node and measures the saved PipelineModel with
`idslib.modeling.get_model_size`, the function ml_07 uses.

Result (2026-10-19, Mac, local[*]): test F1 0.99706 (Table 2: 0.9970), **2.37 MB**.

    env -u SPARK_HOME OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
      JAVA_HOME=$(/usr/libexec/java_home -v 17) IDS_ALLOW_LOCAL_SPARK=1 SPARK_DRIVER_MEMORY=10g \
      python papers/soict2026/results/shap30_xgb_size_20261019/measure_size.py

(pyspark 3.4.1 needs its bundled Spark and JDK 17; the fork-safety flag stops macOS from killing
the XGBoost Python workers.)
