# Papers — FAIR 2026, SOICT 2026 & Journal 2026

Thư mục này chứa **manuscript, figures và kết quả** cho 2 bài báo hội nghị và 1 bài journal.
Code thực thi nằm ở root repo (`shared_utils.py`, `idslib/`, `ml_*.py`, `raspberry/`).

| Thư mục | Nơi nộp | Deadline nộp bài | Trọng tâm |
|---------|---------|------------------|-----------|
| [fair2026/](fair2026/) | [FAIR'2026](https://fair.conf.vn/) | 15/08/2026 | ML, feature selection, SHAP, drift |
| [soict2026/](soict2026/) | [SOICT 2026](https://soict.org/) | 16/09/2026 | Edge phân tán, 2× Jetson Orin Nano Super, Kafka |
| [journal2026/](journal2026/) | Journal | — | Bản mở rộng (ML + edge) |

Luận văn tổng hợp: [../thesis/](../thesis/)

## Script ML (root repo)

| Script | Mô tả | Output |
|--------|-------|--------|
| `ml_00_prepare_cicids2017.py` | Gộp CSV → parquet | `data/` |
| `ml_01_baseline_all_features.py` | Baseline toàn đặc trưng | `results/ml_01_baseline/` |
| `ml_02_feature_selection_rf.py` | RF Top-20/30/40 | `results/ml_02_feature_selection_rf/` |
| `ml_03_hyperparameter_tuning.py` | Grid search (fast mặc định) | `results/ml_03_hyperparameter_tuning/` |
| `ml_04_dimensionality_reduction_pca.py` | PCA k=20/30/40 | `results/ml_04_pca/` |
| `ml_05_shap_explainability.py` | SHAP XAI | `results/ml_05_shap_explainability/` |
| `ml_06_feature_selection_shap.py` | SHAP Top-20/30/40 | `results/ml_06_feature_selection_shap/` |
| `ml_07_cross_method_comparison.py` | So sánh 4 phương pháp + drift | `results/ml_07_cross_method_comparison/` |
| `ml_08_anomaly_gate_autoencoder.py` | Autoencoder gate (edge) | `raspberry/model/` |
| `ml_09_multiclass_eval.py` | Per-attack multiclass + confusion matrix | `results/ml_09_multiclass_eval/` |
| `ml_10_leakage_ablation.py` | Ablation rò rỉ `destination_port` | `results/ml_10_leakage_ablation/` |
| `ml_11_cross_dataset_eval.py` | Tổng quát hóa chéo dataset (2017 ↔ 2018) | `results/ml_11_cross_dataset/` |

Shared: `results/shared/best_config.json` (từ ml_07, dùng cho ml_03)
