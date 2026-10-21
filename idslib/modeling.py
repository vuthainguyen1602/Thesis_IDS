"""IDS modeling: classifiers, training, ensembles, metrics, statistical tests."""
from .core import *
# Hyperparameter tuning utilities (re-exported for ml_03).
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# Uniform class weighting: attack = scale_pos_weight (benign/attack ratio), benign = 1.0.
CLASS_WEIGHT_COL = "classWeight"

_extract_prob_udf = udf(lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0, DoubleType())
_vector_prob_udf = udf(lambda p: Vectors.dense([1.0 - p, p]), VectorUDT())


def add_class_weights(df, scale_pos_weight: float = None,
                      label_col: str = "label_binary",
                      weight_col: str = CLASS_WEIGHT_COL):
    """Attach class-weight column (weightCol) based on benign/attack ratio."""
    if scale_pos_weight is None:
        counts = {r[label_col]: r["count"]
                  for r in df.groupBy(label_col).count().collect()}
        benign, attack = counts.get(0, 0), counts.get(1, 0)
        scale_pos_weight = float(benign) / float(attack) if attack > 0 else 1.0
    spw = float(scale_pos_weight) if scale_pos_weight else 1.0
    return df.withColumn(
        weight_col,
        when(col(label_col) == 1, F.lit(spw)).otherwise(F.lit(1.0)),
    )


def get_classifiers(
    features_col: str,
    label_col: str = "label_binary",
    num_features: int = 50,
    scale_pos_weight: float = None,
    seed: int = 42,
) -> OrderedDict:
    classifiers = OrderedDict()
    w = CLASS_WEIGHT_COL  # weightCol — requires add_class_weights() on train_df

    classifiers["Decision Tree"] = DecisionTreeClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        weightCol=w,
        maxDepth=15,
        minInstancesPerNode=5,
        impurity="entropy",
        seed=seed,
    )

    classifiers["Logistic Regression"] = LogisticRegression(
        featuresCol=features_col,
        labelCol=label_col,
        weightCol=w,
        maxIter=200,
        regParam=0.001,
        elasticNetParam=0.0,
        family="binomial",
        # Standard threshold 0.5 for unbiased comparison across classifiers
    )

    classifiers["SVM (LinearSVC)"] = LinearSVC(
        featuresCol=features_col,
        labelCol=label_col,
        weightCol=w,
        maxIter=200,
        regParam=0.001,
        threshold=0.0,
    )

    classifiers["Naive Bayes"] = NaiveBayes(
        featuresCol=features_col,
        labelCol=label_col,
        weightCol=w,
        modelType="gaussian",
        smoothing=1.0,
    )

    classifiers["Random Forest"] = RandomForestClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        weightCol=w,
        numTrees=200,
        maxDepth=15,
        minInstancesPerNode=5,
        featureSubsetStrategy="sqrt",
        subsamplingRate=1.0,
        seed=seed,
    )

    classifiers["GBT"] = GBTClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        weightCol=w,
        maxIter=150,
        maxDepth=6,
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=seed,
    )

    if HAS_XGBOOST:
        xgb_params = dict(
            features_col=features_col,
            label_col=label_col,
            num_workers=4,
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.05,
            reg_alpha=0.5,
            reg_lambda=1.0,
            use_gpu=False,
        )
        if scale_pos_weight is not None:
            xgb_params["scale_pos_weight"] = scale_pos_weight
        classifiers["XGBoost"] = SparkXGBClassifier(**xgb_params)

    # LightGBM excluded (ARM64 edge target incompatibility)

    layers = [num_features, 64, 32, 2]
    classifiers["MLP"] = MultilayerPerceptronClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        layers=layers,
        maxIter=150,
        blockSize=128,
        stepSize=0.01,
        seed=seed,
    )

    return classifiers


def compute_metrics(
    predictions,
    label_col: str = "label_binary",
) -> dict:
    counts = predictions.groupBy(label_col, "prediction").count().collect()
    TP = TN = FP = FN = 0
    for row in counts:
        l = row[label_col]
        p = row["prediction"]
        c = row["count"]
        if l == 1 and p == 1: TP = c
        elif l == 0 and p == 0: TN = c
        elif l == 0 and p == 1: FP = c
        elif l == 1 and p == 0: FN = c

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    auc_roc = None
    auc_pr = None
    
    score_col = None
    if "rawPrediction" in predictions.columns:
        score_col = "rawPrediction"
    elif "avg_probability" in predictions.columns:
        score_col = "avg_probability"
    elif "probability" in predictions.columns:
        score_col = "probability"

    if score_col:
        try:
            evaluator_roc = BinaryClassificationEvaluator(
                labelCol=label_col, rawPredictionCol=score_col,
                metricName="areaUnderROC"
            )
            auc_roc = evaluator_roc.evaluate(predictions)

            evaluator_pr = BinaryClassificationEvaluator(
                labelCol=label_col, rawPredictionCol=score_col,
                metricName="areaUnderPR"
            )
            auc_pr = evaluator_pr.evaluate(predictions)
        except Exception:
            pass

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }


def print_metrics(metrics: dict, title: str = "") -> None:
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {title}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {metrics['accuracy']:.6f}")
    print(f"  Precision: {metrics['precision']:.6f}")
    print(f"  Recall:    {metrics['recall']:.6f}")
    print(f"  F1-Score:  {metrics['f1']:.6f}")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.6f}")
    if metrics.get("auc_pr") is not None:
        print(f"  AUC-PR:    {metrics['auc_pr']:.6f}")
    print(f"  TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    if metrics.get("model_size_mb") is not None:
        print(f"  Model Size: {metrics['model_size_mb']:.3f} MB")
    if metrics.get("training_time") is not None:
        print(f"  Training time:    {metrics['training_time']:.3f}s")
    if metrics.get("prediction_time") is not None:
        print(f"  Prediction time:  {metrics['prediction_time']:.3f}s")
    print(f"{'=' * 60}")


def get_model_size(model) -> float:
    import shutil
    import tempfile
    # Unique temp dir per call so concurrent get_model_size() calls (e.g. during
    # parallel CrossValidator fits) never delete each other's directory.
    temp_path = tempfile.mkdtemp(prefix="spark_model_size_")
    try:
        shutil.rmtree(temp_path)  # model.save needs the path to not exist yet
        model.save(temp_path)

        total_size = 0
        for dirpath, _dirnames, filenames in os.walk(temp_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

        return total_size / (1024 * 1024)
    except Exception:
        return 0.0
    finally:
        if os.path.exists(temp_path):
            try:
                shutil.rmtree(temp_path)
            except Exception:
                pass


def _get_param(model, param_name: str, default=None):
    """Extract model hyperparameter by getter, paramMap, getOrDefault, or attribute."""
    getter = f"get{param_name[:1].upper()}{param_name[1:]}"
    if hasattr(model, getter):
        try:
            val = getattr(model, getter)()
            if val is not None:
                return val
        except Exception:
            pass

    try:
        pmap = {p.name: v for p, v in model.extractParamMap().items()}
        if param_name in pmap:
            return pmap[param_name]
    except Exception:
        pass

    try:
        val = model.getOrDefault(param_name)
        if val is not None:
            return val
    except Exception:
        pass

    if hasattr(model, param_name):
        val = getattr(model, param_name)
        if "pyspark.ml.param.Param" not in str(type(val)):
            return val

    return default if default is not None else "N/A"


def _get_best_params(cv_model, param_grid):
    best_idx = int(np.argmax(cv_model.avgMetrics))
    best_param_map = param_grid[best_idx]
    return {p.name: v for p, v in best_param_map.items()}


def train_and_evaluate(pipeline, train_df, test_df, title: str = "") -> tuple:
    start = time.time()
    model = pipeline.fit(train_df)
    training_time = time.time() - start

    # Cache and force action so prediction_time captures real latency and compute_metrics reuses predictions
    predictions = model.transform(test_df).cache()
    start_pred = time.time()
    predictions.count()
    prediction_time = time.time() - start_pred

    metrics = compute_metrics(predictions)
    metrics["training_time"] = training_time
    metrics["prediction_time"] = prediction_time
    metrics["model_size_mb"] = get_model_size(model)

    print_metrics(metrics, title)

    predictions.unpersist()
    return model, predictions, metrics


class BaggingModel:
    def __init__(self, models, names=None, weights=None):
        self.models = models
        self.names = names or [f"Model_{i}" for i in range(len(models))]
        if weights is not None:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

    def transform(self, df):
        df_with_id = df.withColumn("_row_id", monotonically_increasing_id()).cache()
        df_with_id.count()
        combined_result = df_with_id.select("_row_id", "label_binary")
        
        for i, model in enumerate(self.models):
            preds = model.transform(df_with_id)
            preds = preds.withColumn(f"prob_{i}", _extract_prob_udf(col("probability")))
            combined_result = combined_result.join(preds.select("_row_id", f"prob_{i}"), on="_row_id")
            
            if (i + 1) % 5 == 0:
                combined_result = combined_result.localCheckpoint()

        weighted_probs = [col(f"prob_{i}") * self.weights[i] for i in range(len(self.models))]
        avg_prob_expr = sum(weighted_probs)
        
        final_res = combined_result.withColumn("avg_probability", avg_prob_expr)
        final_res = final_res.withColumn("rawPrediction", col("avg_probability"))
        final_res = final_res.withColumn("probability", _vector_prob_udf(col("avg_probability")))
        final_res = final_res.withColumn(
            "prediction", 
            when(col("avg_probability") >= 0.5, 1.0).otherwise(0.0)
        )
        return final_res

    def save(self, path: str) -> None:
        """Serialize all bootstrap sub-models for model size measurement and reloading."""
        import json
        os.makedirs(path, exist_ok=True)
        for i, m in enumerate(self.models):
            m.save(os.path.join(path, f"submodel_{i}"))
        with open(os.path.join(path, "bagging_meta.json"), "w") as f:
            json.dump({"names": self.names, "weights": self.weights}, f)

def train_hybrid_bagging(
    pipeline_distribution, train_df,
    label_col: str = "label_binary",
    benign_ratio: float = 0.7, balanced: bool = True,
    feature_list: list = None, feature_subset_rate: float = 1.0,
    seed_base: int = 42,
) -> BaggingModel:
    total_models = sum([count for _, count in pipeline_distribution])
    models = []
    
    if balanced:
        attack_df = train_df.filter(col(label_col) == 1.0).cache()
        benign_df = train_df.filter(col(label_col) == 0.0).cache()
        
        attack_count = attack_df.count()
        benign_count = benign_df.count()
        
        target_benign_count = int((benign_ratio * attack_count) / (1 - benign_ratio))
        benign_fraction = float(target_benign_count) / benign_count
    else:
        full_train_df = train_df.cache()

    current_idx = 0
    for base_pipeline, num_replicas in pipeline_distribution:
        for _ in range(num_replicas):
            current_idx += 1
            print(f"   - Building hybrid model {current_idx}/{total_models} (Balanced={balanced})...")
            
            if balanced:
                attack_sample = attack_df.sample(withReplacement=True, fraction=1.0, seed=seed_base + current_idx)
                benign_sample = benign_df.sample(withReplacement=True, fraction=min(benign_fraction, 10.0), seed=seed_base + 100 + current_idx)
                bag_df = attack_sample.unionAll(benign_sample)
            else:
                bag_df = full_train_df.sample(withReplacement=True, fraction=1.0, seed=seed_base + current_idx)
            
            bag_pipeline = base_pipeline.copy()
            
            if feature_list and feature_subset_rate < 1.0:
                k = max(1, int(len(feature_list) * feature_subset_rate))
                random.seed(seed_base + current_idx)
                selected_features = random.sample(feature_list, k)
                print(f"     * Feature Bagging: {k}/{len(feature_list)} random features selected")
                
                stages = bag_pipeline.getStages()
                for stage in stages:
                    if isinstance(stage, VectorAssembler):
                        stage.setInputCols(selected_features)
                        break
                bag_pipeline.setStages(stages)
            
            model = bag_pipeline.fit(bag_df)
            models.append(model)
            
    if balanced:
        attack_df.unpersist()
        benign_df.unpersist()
    else:
        full_train_df.unpersist()
    
    return BaggingModel(models)


def _train_top3_bagging(
    classifiers, base_stages, results, train_df, test_df, val_df=None, seed: int = 42
):
    base_results = {name: metrics for name, metrics in results.items() 
                    if "Bagging" not in name and "Ensemble" not in name}
    if len(base_results) < 3:
        print("  [WARN] Need at least 3 base models for Hybrid Bagging.")
        return None, None, None

    # Rank by validation F1 (if available) to prevent test-set leakage
    rank_key = "val_f1" if val_df is not None else "f1"
    sorted_names = sorted(base_results.keys(),
                          key=lambda x: base_results[x].get(rank_key, 0), reverse=True)
    top_3 = sorted_names[:3]
    top_3_f1 = [base_results[name].get(rank_key, 0) for name in top_3]
    print(f"  Top-3 models (ranked by {'validation' if val_df is not None else 'test'} F1): {', '.join(top_3)}")

    counts = [3, 2, 2]
    pipeline_dist = []
    total_weights = []
    for i, name in enumerate(top_3):
        clf = classifiers[name]
        pipeline = Pipeline(stages=base_stages + [clf])
        pipeline_dist.append((pipeline, counts[i]))
        for _ in range(counts[i]):
            total_weights.append(top_3_f1[i])

    start_time = time.time()
    ensemble_model = train_hybrid_bagging(
        pipeline_dist, train_df, balanced=False, feature_subset_rate=1.0, seed_base=seed
    )
    ensemble_model.weights = [w / sum(total_weights) for w in total_weights]
    training_time = time.time() - start_time

    start_pred = time.time()
    ens_preds = ensemble_model.transform(test_df)
    prediction_time = time.time() - start_pred
    ens_preds.cache().count()

    metrics = compute_metrics(ens_preds)
    metrics["training_time"] = training_time
    metrics["prediction_time"] = prediction_time

    # Measure serialized size of all sub-models (or fallback to sum of base models)
    real_size = get_model_size(ensemble_model)
    metrics["model_size_mb"] = real_size if real_size and real_size > 0 else sum(
        results[name].get("model_size_mb", 0.5) * count
        for name, count in zip(top_3, counts))

    display_name = "Hybrid Bagging Ensemble (Top-3 Mixed + Weighted)"
    print_metrics(metrics, title=display_name)
    ens_preds.unpersist()
    return display_name, ensemble_model, metrics


def run_all_classifiers(
    assembler,
    scaler,
    train_df,
    test_df,
    features_col: str,
    num_features: int,
    label_col: str = "label_binary",
    extra_stages=None,
    seed: int = 42,
    val_df=None,
) -> tuple:
    """Train base classifiers and Top-3 Hybrid Bagging ensemble."""
    class_counts = train_df.groupBy(label_col).count().collect()
    count_map = {row[label_col]: row["count"] for row in class_counts}
    benign_count = count_map.get(0, 0)
    attack_count = count_map.get(1, 0)
    scale_pos_weight = float(benign_count) / float(attack_count) if attack_count > 0 else 1.0
    print(f"  Class ratio (Benign/Attack): {scale_pos_weight:.4f}")

    # Add class weights for weightCol-supported classifiers and bagging sampler
    train_df = add_class_weights(train_df, scale_pos_weight, label_col)

    classifiers = get_classifiers(
        features_col, label_col, num_features, scale_pos_weight=scale_pos_weight, seed=seed
    )

    base_stages = [assembler, scaler]
    if extra_stages:
        base_stages.extend(extra_stages)

    results = OrderedDict()
    trained_models = OrderedDict()

    for name, clf in classifiers.items():
        print(f"\n{'─' * 60}")
        print(f"  Training: {name}")
        print(f"{'─' * 60}", flush=True)

        try:
            pipeline = Pipeline(stages=base_stages + [clf])
            model, _preds, metrics = train_and_evaluate(
                pipeline, train_df, test_df, title=name
            )
            results[name] = metrics
            trained_models[name] = model
            # Validation F1 used for ensemble selection to prevent test leakage
            if val_df is not None:
                try:
                    v_preds = model.transform(val_df)
                    results[name]["val_f1"] = compute_metrics(
                        v_preds, label_col=label_col)["f1"]
                except Exception as ve:
                    print(f"  [WARN] Validation F1 for {name} failed: {ve}")
                    results[name]["val_f1"] = 0.0
        except Exception as e:
            print(f"  [ERROR] Training {name}: {str(e)}")
            results[name] = {"accuracy": 0, "precision": 0, "recall": 0,
                             "f1": 0, "auc_roc": None, "auc_pr": None,
                             "training_time": 0, "error": str(e)}
            continue

    print(f"\n\n{'=' * 60}")
    print("  TRAINING HYBRID TOP-3 BAGGING ENSEMBLE (3-2-2 WEIGHTED)")
    print("=" * 60)
    try:
        ens_name, ensemble_model, ens_metrics = _train_top3_bagging(
            classifiers, base_stages, results, train_df, test_df, val_df=val_df, seed=seed
        )
        if ens_name:
            results[ens_name] = ens_metrics
            trained_models[ens_name] = ensemble_model
    except Exception as e:
        print(f"  [ERROR] Training Hybrid Bagging: {str(e)}")

    return results, trained_models


def ensemble_voting(
    trained_models: dict,
    test_df,
    results: dict = None,
    label_col: str = "label_binary",
    base_model_names: list = None,
    top_n: int = 3,
) -> dict:
    if base_model_names is None:
        if results is not None:
            base_results = {name: metrics for name, metrics in results.items()
                            if "Bagging" not in name and "Ensemble" not in name and "Voting" not in name}
            # Rank candidates by validation F1 if available, fallback to test F1
            uses_val = any("val_f1" in m for m in base_results.values())
            rank_key = "val_f1" if uses_val else "f1"
            sorted_names = sorted(base_results.keys(),
                                  key=lambda x: base_results[x].get(rank_key, 0), reverse=True)
            base_model_names = [n for n in sorted_names[:top_n] if n in trained_models]
            print(f"  Voting members (ranked by {'validation' if uses_val else 'test'} F1): "
                  f"{', '.join(base_model_names)}")
        else:
            candidates = ["Random Forest", "GBT", "Logistic Regression",
                           "Decision Tree", "XGBoost"]
            base_model_names = [n for n in candidates if n in trained_models][:top_n]

    if len(base_model_names) < 2:
        print("  [WARN] Need at least 2 models for Ensemble Voting")
        return None

    print(f"\n{'─' * 60}")
    print(f"  Ensemble Voting (Top-{len(base_model_names)} by F1): {', '.join(base_model_names)}")
    if results is not None:
        for name in base_model_names:
            f1 = results.get(name, {}).get("f1", 0)
            print(f"    - {name}: F1 = {f1:.6f}")
    print(f"{'─' * 60}")

    start = time.time()
    # Cache with _row_id to prevent non-deterministic re-evaluation across joins
    test_with_id = test_df.withColumn("_row_id", monotonically_increasing_id()).cache()
    test_with_id.count()

    combined = None
    for i, name in enumerate(base_model_names):
        model = trained_models[name]
        preds = model.transform(test_with_id)
        
        if "probability" in preds.columns:
            preds = preds.select(
                "_row_id", label_col,
                _extract_prob_udf(col("probability")).alias(f"prob_{i}")
            )
        else:
            preds = preds.select(
                "_row_id", label_col,
                col("prediction").alias(f"prob_{i}").cast(DoubleType())
            )
            
        combined = preds if combined is None else combined.join(
            preds.select("_row_id", f"prob_{i}"), on="_row_id"
        )

    n = len(base_model_names)
    avg_prob_expr = sum([col(f"prob_{i}") for i in range(n)]) / float(n)
    
    combined = combined.withColumn("avg_probability", avg_prob_expr)
    combined = combined.withColumn("rawPrediction", col("avg_probability"))
    combined = combined.withColumn("probability", _vector_prob_udf(col("avg_probability")))
    combined = combined.withColumn(
        "prediction",
        when(col("avg_probability") >= 0.5, 1.0).otherwise(0.0)
    )

    # Materialize once for accurate latency timing and multi-pass metric caching
    combined = combined.cache()
    combined.count()
    prediction_time = time.time() - start

    metrics = compute_metrics(combined, label_col=label_col)
    total_size = sum(get_model_size(trained_models[name]) for name in base_model_names)

    metrics["training_time"] = 0.0
    metrics["prediction_time"] = prediction_time
    metrics["model_size_mb"] = total_size

    print_metrics(metrics, title=f"Ensemble Voting ({', '.join(base_model_names)})")

    combined.unpersist()
    test_with_id.unpersist()
    return metrics



def print_summary_table(results: dict, title: str = "") -> None:
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    df = pd.DataFrame(results).T
    display_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", 
                    "training_time", "prediction_time", "model_size_mb"]
    available = [c for c in display_cols if c in df.columns]
    for c in available:
        if c not in ["training_time", "prediction_time", "model_size_mb"]:
            df[c] = df[c].apply(lambda x: f"{x:.6f}" if not pd.isna(x) and x is not None else "N/A")
        elif c == "model_size_mb":
            df[c] = df[c].apply(lambda x: f"{x:.3f} MB" if not pd.isna(x) and x is not None else "N/A")
        else:
            df[c] = df[c].apply(lambda x: f"{x:.3f}s" if not pd.isna(x) and x is not None else "N/A")
    print(df[available].to_string())
    print(f"{'=' * 100}")


def _t_critical_95(n: int) -> float:
    """Two-sided 95% t-critical value (df = n - 1) via SciPy or lookup table."""
    if n <= 1:
        return 0.0
    df = n - 1
    try:
        from scipy import stats as _scipy_stats
        return float(_scipy_stats.t.ppf(0.975, df))
    except Exception:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                 15: 2.131, 20: 2.086, 30: 2.042}
        if df in table:
            return table[df]
        # nearest smaller df in table, else normal approx for large df
        keys = [k for k in sorted(table) if k <= df]
        return table[keys[-1]] if keys and df < 40 else 1.96


def summarize_metric_runs(run_metrics: list, metric_keys: list = None,
                          nb_test_train_ratio: float = None) -> dict:
    """Compute mean, std, 95% t-CI, and optional Nadeau-Bengio corrected CI across runs."""
    if metric_keys is None:
        metric_keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]

    if not run_metrics:
        return {}

    summary = {}
    for key in metric_keys:
        vals = [m.get(key) for m in run_metrics if m.get(key) is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        n = len(arr)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        # Exact t-distribution critical value for small sample sizes
        t_crit = _t_critical_95(n)
        ci95_half = float(t_crit * std_val / np.sqrt(n)) if n > 1 else 0.0
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
        summary[f"{key}_ci95_low"] = mean_val - ci95_half
        summary[f"{key}_ci95_high"] = mean_val + ci95_half
        if nb_test_train_ratio is not None and n > 1:
            nb_half = float(t_crit * std_val *
                            np.sqrt(1.0 / n + float(nb_test_train_ratio)))
            summary[f"{key}_ci95_nb_low"] = mean_val - nb_half
            summary[f"{key}_ci95_nb_high"] = mean_val + nb_half
        summary[f"{key}_n"] = len(arr)
    return summary


def paired_cohens_d(scores_a: list, scores_b: list) -> float:
    """Cohen's d effect size for paired samples: mean(diff) / std(diff, ddof=1)."""
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return 0.0
    diffs = np.array(scores_a, dtype=float) - np.array(scores_b, dtype=float)
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diffs)) / sd


def tost_equivalence(scores_a: list, scores_b: list, margin: float = 0.001,
                     nb_test_train_ratio: float = None, alpha: float = 0.05) -> dict:
    """Two One-Sided Tests (TOST) for practical equivalence within margin.
    
    Supports Nadeau-Bengio SE correction for overlapping resamples.
    """
    out = {"margin": float(margin), "mean_diff": 0.0, "se": 0.0,
           "p_tost": None, "equivalent": False, "min_margin": None}
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return out

    diffs = np.array(scores_a, dtype=float) - np.array(scores_b, dtype=float)
    n = len(diffs)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd * float(np.sqrt(1.0 / n + nb_test_train_ratio)) if nb_test_train_ratio \
        else sd / float(np.sqrt(n))
    out.update({"mean_diff": mean, "se": se})
    if se == 0.0:
        out.update({"p_tost": 0.0, "equivalent": abs(mean) < margin, "min_margin": abs(mean)})
        return out

    try:
        from scipy import stats as _scipy_stats
        df = n - 1
        p_lower = float(_scipy_stats.t.sf((mean + margin) / se, df))   # H0: diff <= -margin
        p_upper = float(_scipy_stats.t.cdf((mean - margin) / se, df))  # H0: diff >= +margin
        p_tost = max(p_lower, p_upper)
        t_one_sided = float(_scipy_stats.t.ppf(1.0 - alpha, df))
        out.update({"p_tost": p_tost, "equivalent": p_tost < alpha,
                    "min_margin": abs(mean) + t_one_sided * se})
    except Exception:
        # scipy missing: skip the p-value rather than approximate it
        pass
    return out


def permutation_pvalue(scores_a: list, scores_b: list, n_permutations: int = 2000, seed: int = 42) -> float:
    """Two-sided paired sign-permutation p-value (exact enumeration for small N, Monte Carlo otherwise)."""
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return 1.0

    arr_a = np.array(scores_a, dtype=float)
    arr_b = np.array(scores_b, dtype=float)
    diffs = arr_a - arr_b
    observed = abs(float(np.mean(diffs)))
    n = len(diffs)

    if 2 ** n <= max(n_permutations, 4096):
        # Exact enumeration of all 2^n sign patterns
        extreme = 0
        total = 2 ** n
        for mask in range(total):
            signs = np.array([1.0 if (mask >> j) & 1 else -1.0 for j in range(n)])
            perm_stat = abs(float(np.mean(diffs * signs)))
            if perm_stat >= observed - 1e-15:
                extreme += 1
        return extreme / total

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        perm_stat = abs(float(np.mean(diffs * signs)))
        if perm_stat >= observed:
            extreme += 1
    return (extreme + 1.0) / (n_permutations + 1.0)

