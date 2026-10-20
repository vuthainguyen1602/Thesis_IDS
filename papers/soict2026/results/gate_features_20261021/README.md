# Gate feature set: all 77 leak-free features vs. the classifier's SHAP Top-30 (2026-10-21)

Source of the SOICT statement (Gate trade-off paragraph) that the gate forwards
68.1% of attack flows on all 77 leak-free features but only 44.8% when it scores
the classifier's 30 SHAP features, at the same pre-committed quantile.

`run.py` repeats the gate's training procedure from `ml_08_anomaly_gate_autoencoder.py`
(benign 90/10 fit/validation split, seed 42, MLP autoencoder 64-16-64, threshold at
the 0.995 quantile of the validation benign MSE) once per feature set and scores the
CICIDS2017 test split (380,119 benign / 67,156 attack). It writes nothing under
`jetson/model`; the deployed gate is the 77-feature model.

| Feature set | Recall (attack forwarded) | FPR | Gate skip |
|---|---|---|---|
| all 77 leak-free | 68.07% | 0.486% | 89.37% |
| SHAP Top-30 | 44.78% | 0.490% | 92.86% |

The 77-feature row reproduces `results/ml_08_anomaly_gate/gate_operating_points.csv`
(q = 0.995) to every digit. Full rows: `gate_feature_sets.csv`; console: `run.log`.
