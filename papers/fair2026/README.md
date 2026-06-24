# FAIR'2026 — Paper: ML & Feature Selection for IDS

**Venue:** [The 19th FAIR — Fundamental and Applied IT Research](https://fair.conf.vn/)
**Topic:** Artificial Intelligence and Its Future Trends
**Deadline:** 2026-08-15 | **Conference:** 2026-10-08/09, HUIT, Ho Chi Minh City

## Research question

A systematic comparison of dimensionality-reduction and feature-selection methods (RF Importance, SHAP, PCA) for network intrusion detection on Apache Spark ML, evaluated on CICIDS2017 under a leakage-aware protocol.

## Title

**VI:** *So sánh giảm chiều và lựa chọn đặc trưng cho phát hiện xâm nhập mạng trên Apache Spark ML với quy trình đánh giá loại trừ rò rỉ dữ liệu*
**EN:** *A Comparative Study of Feature Selection and Dimensionality Reduction for Network Intrusion Detection on Spark ML under a Leakage-Aware Protocol*

## Experiments (code at repo root)

| Script | Description |
|--------|-------------|
| `ml_01_baseline_all_features.py` | Baseline — all features |
| `ml_02_feature_selection_rf.py` | RF Top-20/30/40 |
| `ml_04_dimensionality_reduction_pca.py` | PCA k=20/30/40 |
| `ml_05_shap_explainability.py` | SHAP XAI |
| `ml_06_feature_selection_shap.py` | SHAP Top-20/30/40 |
| `ml_07_cross_method_comparison.py` | 4-method comparison + drift/robustness |
| `ml_09_multiclass_eval.py` | Per-attack multiclass + confusion matrix |
| `ml_10_leakage_ablation.py` | `destination_port` leakage ablation |

## Reproduce

All experiments run on the distributed cluster (Mac + 2× Jetson). See [../../cluster/DISTRIBUTED_CLUSTER.md](../../cluster/DISTRIBUTED_CLUSTER.md).

```bash
export IDS_ROOT="$(pwd)"
./cluster/reproduce_cluster.sh fair     # dispatches ml_01..ml_10 to the Jetson workers
./cluster/pull_results.sh
./papers/fair2026/collect_results.sh
```

## Main figures

Figures are read directly from the results folders (no duplicate copies):

| Figure | Source |
|--------|--------|
| F1 heatmap | `results/ml_07_cross_method_comparison/f1_heatmap.png` |
| Cross-method F1 | `results/ml_07_cross_method_comparison/cross_method_f1_comparison.png` |
| Train/predict time | `results/ml_07_cross_method_comparison/train_predict_time.png` |
| Confusion matrix | `results/ml_09_multiclass_eval/confusion_matrix.png` |
| Leakage ablation | `results/ml_10_leakage_ablation/leakage_ablation.png` |

## Manuscript

The paper is in `manuscript/` (IEEE conference template, XeLaTeX + Vietnamese).

## Out of scope for this paper

- Jetson Orin Nano Super / Kafka deployment → see [../soict2026/](../soict2026/)
- Full thesis → see [../../thesis/](../../thesis/)
