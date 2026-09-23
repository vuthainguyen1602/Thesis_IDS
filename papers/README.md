# Papers — FAIR 2026 & SOICT 2026

This folder holds the **manuscripts, figures, and results** for two conference papers. The execution code lives at the repo root (`shared_utils.py`, `idslib/`, `ml_*.py`, `jetson/`).

| Folder | Venue | Submission deadline | Focus |
|--------|-------|---------------------|-------|
| [fair2026/](fair2026/) | [FAIR'2026](https://fair.conf.vn/) | 2026-08-15 | ML, feature selection, SHAP, drift |
| [soict2026/](soict2026/) | [SOICT 2026](https://soict.org/) | 2026-09-16 | Distributed edge, 2× Jetson Orin Nano Super, Kafka |

Combined thesis: [../thesis/](../thesis/)

## ML scripts (repo root)

| Script | Description | Output |
|--------|-------------|--------|
| `ml_00_prepare_cicids2017.py` | Merge CSV → parquet | `data/` |
| `ml_01_baseline_all_features.py` | Baseline (all features) | `results/ml_01_baseline/` |
| `ml_02_feature_selection_rf.py` | RF Top-20/30/40 | `results/ml_02_feature_selection_rf/` |
| `ml_03_hyperparameter_tuning.py` | Grid search (fast by default) | `results/ml_03_hyperparameter_tuning/` |
| `ml_04_dimensionality_reduction_pca.py` | PCA k=20/30/40 | `results/ml_04_pca/` |
| `ml_05_shap_explainability.py` | SHAP XAI | `results/ml_05_shap_explainability/` |
| `ml_06_feature_selection_shap.py` | SHAP Top-20/30/40 | `results/ml_06_feature_selection_shap/` |
| `ml_07_cross_method_comparison.py` | 4-method comparison + drift | `results/ml_07_cross_method_comparison/` |
| `ml_08_anomaly_gate_autoencoder.py` | Autoencoder gate (edge) + operating curve | `jetson/model/anomaly_*.pkl`, `results/ml_08_anomaly_gate/` |
| `ml_09_multiclass_eval.py` | Per-attack multiclass + confusion matrix | `results/ml_09_multiclass_eval/` |
| `ml_10_leakage_ablation.py` | `destination_port` leakage ablation | `results/ml_10_leakage_ablation/` |
| `ml_11_cross_dataset_eval.py` | Cross-dataset generalization (2017 ↔ 2018) | `results/ml_11_cross_dataset/` |

Shared: `results/shared/best_config.json` (written by ml_07, consumed by ml_03).

> The cluster of 1 Mac + 2 Jetson Orin Nano Super is the measured setup, and the one the papers report:
> see [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md) and
> [../jetson/JETSON_DISTRIBUTED.md](../jetson/JETSON_DISTRIBUTED.md). It is not the only way to run the
> code: `IDS_ALLOW_LOCAL_SPARK=1` lets the offline pipeline reproduce on a single machine (as the FAIR
> manuscript states), and edge Mode A runs on a `local[*]` session — only the Mode C benchmark needs
> the standalone cluster.
