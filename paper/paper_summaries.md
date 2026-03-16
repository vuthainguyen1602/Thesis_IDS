# IDS Research Publication Roadmap (Q1)

This document provides a technical summary of the two papers extracted from the `Thesis_IDS` project, highlighting their research novelty and key results.

---

## Paper 1: Performance & Hybrid Architecture
**Target Title:** *A Heterogeneous Hybrid Ensemble Learning Framework with Class-Aware Sampling for Robust Network Intrusion Detection*

### 🚀 Research Novelty
- **Heterogeneous Ensemble**: Unlike standard Random Forest (homogeneous), this paper proposes merging **XGBoost** (Boosting), **LightGBM** (Leaf-wise growth), and **MLP** (Neural Networks) using soft-voting.
- **Class-Aware Balanced Sampling**: A custom bootstrap method that dynamically balances the CICIDS2017 dataset during ensemble training, specifically improving the recall of rare attacks (e.g., Infiltration, Heartbleed).

### 📊 Key Results
- **Overall F1-Score**: 0.9994 (State-of-the-Art for CICIDS2017).
- **Inference Stability**: Demonstrated lower variance across cross-validation folds compared to individual models.
- **Spark Optimization**: Effectively utilizes PySpark's local[*] master for parallel ensemble training.

---

## Paper 2: Explainability & Edge Intelligence
**Target Title:** *X-IDS: Toward Explainable and Resource-Efficient Intrusion Detection at the Edge using SHAP and Distributed Stream Processing*

### 🚀 Research Novelty
- **SHAP-Driven Input Pruning**: First use of SHAP (Shapley Additive Explanations) not just for "explaining" but as an **operational filter** to reduce feature dimensionality for edge deployment.
- **Edge Stream Processing**: Real-time deployment on **Raspberry Pi 4** using a full production-grade stack (Kafka $\rightarrow$ PySpark $\rightarrow$ InfluxDB/Grafana).
- **Trade-off Analysis**: Quantitative study of "Inference Latency vs. Metric Integrity" when moving from cloud to edge hardware.

### 📊 Key Results
- **Model Efficiency**: Reduced feature set from 78 to 30 while keeping F1-score at 0.9918.
- **Resource Footprint**: Decision Tree deployment consumed only **30% RAM** on Raspberry Pi, leaving room for other edge services.
- **End-to-End Latency**: Achieved ~100ms per sample inference in a real-time Kafka stream.

---

## 📝 Publication Advice
1. **Paper 1** is more "algorithmic." Target *IEEE Transactions on Information Forensics and Security* or *IEEE Access*.
2. **Paper 2** is more "systems/IoT." Target *IEEE Internet of Things Journal* or *IEEE Transactions on Dependable and Secure Computing*.
3. **LaTeX Usage**: Use the generated `.tex` files. You will need the `IEEEtran.cls` file (standard in most LaTeX distros or Overleaf).
