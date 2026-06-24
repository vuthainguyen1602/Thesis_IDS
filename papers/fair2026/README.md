# FAIR'2026 — Paper: ML & Feature Selection for IDS

**Hội nghị:** [The 19th FAIR — Fundamental and Applied IT Research](https://fair.conf.vn/)  
**Chủ đề:** Artificial Intelligence and Its Future Trends  
**Deadline:** 15/08/2026 | **Hội nghị:** 08–09/10/2026, HUIT TP.HCM

## Câu hỏi nghiên cứu

So sánh có hệ thống các phương pháp giảm chiều (RF Importance, SHAP, PCA) cho phát hiện xâm nhập mạng trên Apache Spark ML với dataset CICIDS2017.

## Đề xuất tiêu đề

**VI:** *So sánh các phương pháp giảm chiều và lựa chọn đặc trưng cho phát hiện xâm nhập mạng sử dụng Apache Spark ML*

**EN:** *A Comparative Study of Feature Selection and Dimensionality Reduction for Network Intrusion Detection using Spark ML*

## Experiments (code ở root repo)

| Script | Mô tả |
|--------|-------|
| `ml_01_baseline_all_features.py` | Baseline — all features |
| `ml_02_feature_selection_rf.py` | RF Top-20/30/40 |
| `ml_04_dimensionality_reduction_pca.py` | PCA k=20/30/40 |
| `ml_05_shap_explainability.py` | SHAP XAI |
| `ml_06_feature_selection_shap.py` | SHAP Top-20/30/40 |
| `ml_07_cross_method_comparison.py` | So sánh 4 phương pháp + drift/robustness |

## Reproduce

```bash
export IDS_ROOT="$(pwd)"
./papers/fair2026/reproduce.sh
./papers/fair2026/collect_results.sh
```

## Figures chính cho bài báo

Hình lấy trực tiếp từ thư mục gốc (không copy trùng):

| Figure | Nguồn |
|--------|-------|
| F1 heatmap | `results/ml_07_cross_method_comparison/f1_heatmap.png` |
| Cross-method F1 | `results/ml_07_cross_method_comparison/cross_method_f1_comparison.png` |
| Drift simulation | `results/ml_07_cross_method_comparison/drift_simulation_f1.png` |

## Manuscript

Đặt bài viết trong `manuscript/` (LaTeX template FAIR hoặc Word theo hướng dẫn hội nghị).

## Không nằm trong paper này

- Triển khai Jetson Orin Nano Super / Kafka → xem [../soict2026/](../soict2026/)
- Luận văn đầy đủ → xem [../../thesis/](../../thesis/)
