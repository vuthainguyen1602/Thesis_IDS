"""Retrain SHAP Top-30 + XGBoost (ml_07 config) on one node and measure its saved size."""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
os.environ.pop("SPARK_MASTER", None)
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
import ml_07_cross_method_comparison as m7
from idslib.core import create_spark_session
from idslib import modeling as md

spark = create_spark_session("SHAP30_XGB_size")
df, train_df, test_df, feature_cols = m7.load_and_prepare_data(spark)
train_core, val_df = train_df.randomSplit([0.8, 0.2], seed=42)
cfg = m7.METHODS["SHAP Top-30"]
feats = pd.read_csv(cfg["csv"]).head(cfg["top_k"])[cfg["col"]].tolist()
label = "label_binary"
benign = train_core.filter(f"{label} = 0").count(); attack = train_core.filter(f"{label} = 1").count()
spw = benign / attack
clf = md.get_classifiers("features_scaled", label, 30, scale_pos_weight=spw)["XGBoost"]
pipe = Pipeline(stages=[VectorAssembler(inputCols=feats, outputCol="features_raw", handleInvalid="keep"),
                        StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=True), clf])
t = time.time(); model = pipe.fit(train_core); tt = time.time() - t
pred = model.transform(test_df).select(label, "prediction").toPandas()
from sklearn.metrics import f1_score
f1 = f1_score(pred[label], pred["prediction"])
size = md.get_model_size(model)
out = {"f1_test": f1, "model_size_mb": size, "train_s": tt, "spw": spw, "n_features": len(feats)}
print("RESULT", json.dumps(out))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json"), "w"))
spark.stop()
