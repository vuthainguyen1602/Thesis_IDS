# SHAP Top-30 ranked on Random Forest vs. on XGBoost

Question: the thesis/SOICT SHAP Top-30 list comes from TreeExplainer on the 77-feature XGBoost
model (ml_06), but the deployed classifier is a Random Forest. Would SHAP computed on the Random
Forest itself pick a better list?

`run.py` ranks with SHAP on the ml_07 Random Forest (77 features, full train split, 2,059-row
stratified train sample in the fitted scaler's space, attack-class SHAP), keeps the Top-30
(`shap_rf_feature_importance.csv`), then trains four tree models on each list under the ml_07
protocol (train_core/val seed 42, class weights, get_classifiers parameters) and scores test F1.
Both lists ran in the same session (2026-10-20, Mac, local[*]).

| Model | SHAP-RF Top-30 | SHAP-XGBoost Top-30 |
|---|---|---|
| Random Forest | 0.9837 | **0.9954** |
| XGBoost | 0.9942 | **0.9971** |
| Decision Tree | 0.9834 | **0.9948** |
| GBT | 0.9797 | **0.9916** |

The SHAP-XGBoost rerun matches ml_07 (RF 0.9955, XGBoost 0.9970, DT 0.9948, GBT 0.9918).

Top-30 overlap: SHAP-RF vs RF importance 25, SHAP-RF vs SHAP-XGBoost 19, SHAP-XGBoost vs RF
importance 18. The ranking model matters more than SHAP vs impurity importance.

The 11 features only the RF list keeps are near-duplicates of features both lists share
(subflow_* counts, packet_length_variance, avg_fwd_segment_size, fwd_header_length55,
max_packet_length). The 11 only the XGBoost list keeps are timing and flag features (idle_min/max,
flow_iat_min/max, fwd_iat_min, active_min, fin/psh_flag_count, min_seg_size_forward,
act_data_pkt_fwd, fwd_packet_length_mean). Random Forest spreads credit across correlated
copies; boosting credits one copy and moves on, so its Top-30 covers more distinct signals.

Single split, no resampling: the 0.003-0.012 gaps are at least 10x the 0.0001-0.0002 cluster-vs-local spread of the
rerun above, but they carry no p-value.

    env -u SPARK_HOME OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
      JAVA_HOME=$(/usr/libexec/java_home -v 17) IDS_ALLOW_LOCAL_SPARK=1 SPARK_DRIVER_MEMORY=10g \
      python papers/soict2026/results/shap_rf_vs_xgb_20261020/run.py

Runtime about 45 min (RF fit on 77 features 11 min, SHAP on 200 trees 2 min).
