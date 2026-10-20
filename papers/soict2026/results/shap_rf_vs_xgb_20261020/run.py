"""SHAP Top-30 ranked on Random Forest vs. on XGBoost (ml_06 ranking, ml_07 evaluation).

Ranking (as ml_06, but with the ml_07 Random Forest instead of XGBoost): fit RF on all
77 features over the full train split, explain a 2,000-row class-stratified train sample
with shap.TreeExplainer in the fitted scaler's space, keep the Top-30 by mean |SHAP|.

Evaluation (as ml_07): train_core/val split seed 42, class weights, get_classifiers
parameters, test F1 on the untouched test split. Both lists are re-run here so the
comparison shares one environment.
"""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
os.environ.pop("SPARK_MASTER", None)
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import f1_score
import ml_07_cross_method_comparison as m7
from idslib.core import create_spark_session
from idslib import modeling as md
from idslib.data import stratified_sample

OUT = os.path.dirname(os.path.abspath(__file__))
LABEL = "label_binary"
MODELS = [m for m in os.environ.get("MODELS", "Random Forest,XGBoost,Decision Tree,GBT").split(",") if m]

spark = create_spark_session("SHAP_RF_vs_XGB")
df, train_df, test_df, feature_cols = m7.load_and_prepare_data(spark)


def spw_of(d):
    c = {r[LABEL]: r["count"] for r in d.groupBy(LABEL).count().collect()}
    return c[0] / c[1]


def prep(feats):
    return [VectorAssembler(inputCols=feats, outputCol="features_raw", handleInvalid="keep"),
            StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True)]


# ---- 1. SHAP ranking on Random Forest -------------------------------------------------
rank_csv = os.path.join(OUT, "shap_rf_feature_importance.csv")
if not os.path.exists(rank_csv):
    import shap
    spw = spw_of(train_df)
    rf = md.get_classifiers("features_scaled", LABEL, len(feature_cols), scale_pos_weight=spw)["Random Forest"]
    t = time.time()
    model = Pipeline(stages=prep(feature_cols) + [rf]).fit(md.add_class_weights(train_df, spw))
    print(f"RF 77-feature fit {time.time() - t:.0f}s", flush=True)
    asm, scl, rf_model = model.stages
    sample = stratified_sample(train_df, feature_cols + [LABEL], LABEL, 2000)
    X = np.array(scl.transform(asm.transform(sample))
                 .select(vector_to_array("features_scaled").alias("x")).toPandas()["x"].tolist(),
                 dtype=np.float64)
    t = time.time()
    explainer = shap.TreeExplainer(rf_model)
    print(f"TreeExplainer built {time.time() - t:.0f}s", flush=True)
    t = time.time()
    sv = np.asarray(explainer.shap_values(X, check_additivity=False))
    print(f"SHAP values {sv.shape} {time.time() - t:.0f}s", flush=True)
    if sv.ndim == 3:  # per-class output: keep the attack class
        sv = sv[1] if sv.shape[0] == 2 else sv[..., 1]
    pd.DataFrame({"feature": feature_cols, "shap_importance": np.abs(sv).mean(axis=0)}) \
        .sort_values("shap_importance", ascending=False).to_csv(rank_csv, index=False)

rf_top = pd.read_csv(rank_csv).head(30)["feature"].tolist()
xgb_top = pd.read_csv(m7.SHAP_IMPORTANCE_CSV).head(30)["feature"].tolist()
imp_top = pd.read_csv(m7.RF_IMPORTANCE_CSV).head(30)["feature"].tolist()
overlap = {"shap_rf_vs_shap_xgb": len(set(rf_top) & set(xgb_top)),
           "shap_rf_vs_rf_importance": len(set(rf_top) & set(imp_top)),
           "shap_xgb_vs_rf_importance": len(set(xgb_top) & set(imp_top))}
print("OVERLAP", overlap, flush=True)

# ---- 2. Evaluate both lists under the ml_07 protocol ----------------------------------
train_core, val_df = train_df.randomSplit([0.8, 0.2], seed=42)
train_core = train_core.cache()
spw = spw_of(train_core)
train_w = md.add_class_weights(train_core, spw)
res_json = os.path.join(OUT, "result.json")
rows = json.load(open(res_json))["rows"] if os.path.exists(res_json) else []
done = {(r["list"], r["model"]) for r in rows}
for list_name, feats in [("SHAP-RF Top-30", rf_top), ("SHAP-XGBoost Top-30", xgb_top)]:
    for name in MODELS:
        if (list_name, name) in done:
            continue
        clf = md.get_classifiers("features_scaled", LABEL, 30, scale_pos_weight=spw)[name]
        t = time.time()
        model = Pipeline(stages=prep(feats) + [clf]).fit(train_w)
        tt = time.time() - t
        pred = model.transform(test_df).select(LABEL, "prediction").toPandas()
        row = {"list": list_name, "model": name, "f1_test": f1_score(pred[LABEL], pred["prediction"]),
               "train_s": tt}
        print("RESULT", json.dumps(row), flush=True)
        rows.append(row)
        json.dump({"overlap": overlap, "shap_rf_top30": rf_top, "rows": rows}, open(res_json, "w"), indent=1)
spark.stop()
