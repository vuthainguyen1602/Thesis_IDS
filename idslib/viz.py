"""IDS visualisation: comparison plots and SHAP explainability."""
from .core import *  # noqa: F401,F403
from .data import stratified_sample  # noqa: F401


def plot_comparison(
    results: dict, title: str = "Algorithm Comparison",
    save_path: str = None, show: bool = True,
) -> None:
    names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    _fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, metric, color in zip(axes.flatten(), metric_names, colors):
        values = [results[n].get(metric, 0) for n in names]
        bars = ax.barh(names, values, color=color, alpha=0.85)
        ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
        ax.set_xlim(min(values) - 0.02 if min(values) > 0.02 else 0, 1.005)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_training_time(
    results: dict, title: str = "Training Time Comparison",
    save_path: str = None, show: bool = True,
) -> None:
    names = list(results.keys())
    times = [results[n].get("training_time", 0) for n in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, times, color=colors)
    for bar, val in zip(bars, times):
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}s", va="center", fontsize=9)
    plt.xlabel("Time (seconds)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_time(
    results: dict, title: str = "Prediction Time Comparison",
    save_path: str = None, show: bool = True,
) -> None:
    names = list(results.keys())
    times = [results[n].get("prediction_time", 0) for n in names]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, times, color=colors)
    for bar, val in zip(bars, times):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}s", va="center", fontsize=9)
    plt.xlabel("Time (seconds)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_size(
    results: dict, title: str = "Model Size Comparison",
    save_path: str = None, show: bool = True,
) -> None:
    names = list(results.keys())
    sizes = [results[n].get("model_size_mb", 0) for n in names]
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(names)))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, sizes, color=colors)
    for bar, val in zip(bars, sizes):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f} MB", va="center", fontsize=9)
    plt.xlabel("Size (MB)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices(
    results: dict, title: str = "Confusion Matrices",
    save_path: str = None, show: bool = True,
) -> None:
    valid = {k: v for k, v in results.items()
             if all(key in v for key in ["TP", "TN", "FP", "FN"]) and v.get("TP", 0) + v.get("TN", 0) > 0}
    if not valid:
        print("  [WARN] No confusion matrix data available.")
        return

    n = len(valid)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    _fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))

    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, metrics) in enumerate(valid.items()):
        ax = axes[idx]
        cm = np.array([[metrics["TN"], metrics["FP"]],
                        [metrics["FN"], metrics["TP"]]])

        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Benign", "Attack"],
                    yticklabels=["Benign", "Attack"],
                    cbar=False, annot_kws={"fontsize": 10})
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        display_name = name if len(name) <= 25 else name[:22] + "..."
        ax.set_title(display_name, fontsize=10, fontweight="bold")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curves(
    trained_models: dict, test_df,
    label_col: str = "label_binary",
    title: str = "ROC Curves",
    save_path: str = None, show: bool = True,
) -> None:
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import DoubleType
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    extract_prob = udf(lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0, DoubleType())

    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(trained_models), 10)))

    for idx, (name, model) in enumerate(trained_models.items()):
        try:
            preds = model.transform(test_df)

            if "probability" not in preds.columns:
                print(f"  [WARN] Skipping {name}: no probability column")
                continue

            prob_df = preds.select(
                col(label_col).alias("label"),
                extract_prob(col("probability")).alias("prob_pos"),
            ).toPandas()

            labels = prob_df["label"].values
            probs = prob_df["prob_pos"].values

            valid_mask = np.isfinite(probs) & np.isfinite(labels)
            labels = labels[valid_mask]
            probs = probs[valid_mask]

            if len(labels) == 0:
                continue

            total_pos = np.sum(labels == 1)
            total_neg = np.sum(labels == 0)

            if total_pos == 0 or total_neg == 0:
                continue

            # Keep AUC computation consistent with table metrics.
            auc_val = None
            score_col = None
            if "rawPrediction" in preds.columns:
                score_col = "rawPrediction"
            elif "avg_probability" in preds.columns:
                score_col = "avg_probability"
            elif "probability" in preds.columns:
                score_col = "probability"

            if score_col is not None:
                try:
                    evaluator_roc = BinaryClassificationEvaluator(
                        labelCol=label_col,
                        rawPredictionCol=score_col,
                        metricName="areaUnderROC",
                    )
                    auc_val = evaluator_roc.evaluate(preds)
                except Exception:
                    auc_val = None

            # ROC curve points from unique thresholds (descending score order).
            order = np.argsort(-probs, kind="mergesort")
            labels_sorted = labels[order]
            probs_sorted = probs[order]

            distinct_idx = np.where(np.diff(probs_sorted))[0]
            threshold_idx = np.r_[distinct_idx, labels_sorted.size - 1]

            tps = np.cumsum(labels_sorted == 1)[threshold_idx]
            fps = (threshold_idx + 1) - tps

            tpr_sorted = np.r_[0.0, tps / total_pos]
            fpr_sorted = np.r_[0.0, fps / total_neg]

            if auc_val is None:
                if hasattr(np, "trapezoid"):
                    auc_val = np.trapezoid(tpr_sorted, fpr_sorted)
                elif hasattr(np, "trapz"):
                    auc_val = np.trapz(tpr_sorted, fpr_sorted)
                else:
                    auc_val = np.sum((fpr_sorted[1:] - fpr_sorted[:-1]) * (tpr_sorted[1:] + tpr_sorted[:-1]) / 2)

            color = colors[idx % len(colors)]
            display_name = name if len(name) <= 20 else name[:17] + "..."
            plt.plot(fpr_sorted, tpr_sorted, color=color, linewidth=1.5,
                     label=f"{display_name} (AUC={auc_val:.6f})")

        except Exception as e:
            print(f"  [WARN] Skipping {name}: {str(e)}")
            continue

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random (AUC=0.5)")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()



def shap_explain_model(
    spark_model, df_to_explain, feature_cols: list, output_dir: str,
    sample_size: int = 1000, label_col: str = "label_binary",
) -> dict:
    import shap
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = {}
    
    print(f"\n{'─' * 60}")
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print(f"{'─' * 60}")
    
    print(f"  [1/5] Collecting {sample_size} samples to Pandas for explanation...")
    
    select_cols = feature_cols + [label_col]
    sample_df = stratified_sample(df_to_explain, select_cols, label_col, sample_size)
    pdf = sample_df.toPandas()
    
    X_explain = pdf[feature_cols].values
    y_explain = pdf[label_col].values
    
    print(f"        Collected: {X_explain.shape[0]} samples, {X_explain.shape[1]} features")
    print(f"        Attack ratio: {y_explain.mean():.2%}")
    
    print("  [2/5] Extracting XGBoost model from PipelineModel...")
    
    xgb_model = None
    for stage in spark_model.stages:
        stage_class = type(stage).__name__
        if "XGBoost" in stage_class or "XGB" in stage_class:
            xgb_model = stage
            break
    
    if xgb_model is None:
        print("  [ERROR] No XGBoost stage found in pipeline. SHAP requires XGBoost.")
        print("  Available stages:", [type(s).__name__ for s in spark_model.stages])
        return saved_plots
    
    booster = xgb_model.get_booster()
    print(f"        Extracted booster: {booster.num_boosted_rounds()} rounds")
    
    print("  [3/5] Computing SHAP values (TreeExplainer)...")
    
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(X_explain)
    
    shap_values.feature_names = feature_cols
    
    print(f"        SHAP values computed: shape {shap_values.values.shape}")
    
    print("  [4/5] Generating SHAP plots...")
    
    plt.figure(figsize=(12, 10))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Summary Plot - Feature Impact on Attack Detection", fontsize=14, pad=20)
    plt.tight_layout()
    path_summary = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(path_summary, dpi=200, bbox_inches='tight')
    plt.close()
    saved_plots["summary_beeswarm"] = path_summary
    print(f"        [INFO] Saved: {path_summary}")
    
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("SHAP Global Feature Importance (Top-20)", fontsize=14, pad=20)
    plt.tight_layout()
    path_bar = os.path.join(output_dir, "shap_feature_importance_bar.png")
    plt.savefig(path_bar, dpi=200, bbox_inches='tight')
    plt.close()
    saved_plots["importance_bar"] = path_bar
    print(f"        [INFO] Saved: {path_bar}")
    
    attack_indices = np.where(y_explain == 1)[0]
    if len(attack_indices) > 0:
        attack_idx = attack_indices[0]
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[attack_idx], max_display=15, show=False)
        plt.title(f"SHAP Waterfall - Why Sample #{attack_idx} Was Classified as Attack", 
                  fontsize=12, pad=20)
        plt.tight_layout()
        path_wf_attack = os.path.join(output_dir, "shap_waterfall_attack.png")
        plt.savefig(path_wf_attack, dpi=200, bbox_inches='tight')
        plt.close()
        saved_plots["waterfall_attack"] = path_wf_attack
        print(f"        [INFO] Saved: {path_wf_attack}")
    
    benign_indices = np.where(y_explain == 0)[0]
    if len(benign_indices) > 0:
        benign_idx = benign_indices[0]
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[benign_idx], max_display=15, show=False)
        plt.title(f"SHAP Waterfall - Why Sample #{benign_idx} Was Classified as Benign", 
                  fontsize=12, pad=20)
        plt.tight_layout()
        path_wf_benign = os.path.join(output_dir, "shap_waterfall_benign.png")
        plt.savefig(path_wf_benign, dpi=200, bbox_inches='tight')
        plt.close()
        saved_plots["waterfall_benign"] = path_wf_benign
        print(f"        [INFO] Saved: {path_wf_benign}")
    
    print("  [5/5] Generating SHAP feature ranking table...")
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_SHAP_Value": mean_abs_shap
    }).sort_values("Mean_SHAP_Value", ascending=False)
    
    csv_path = os.path.join(output_dir, "shap_feature_importance.csv")
    feature_importance.to_csv(csv_path, index=False)
    saved_plots["importance_csv"] = csv_path
    
    print(f"\n  {'─' * 50}")
    print(f"  Top-20 Features by Mean |SHAP Value|")
    print(f"  {'─' * 50}")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    print(f"  {'─' * 50}")
    for rank, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        print(f"  {rank:<6} {row['Feature']:<35} {row['Mean_SHAP_Value']:>12.6f}")
    
    print(f"\n  SHAP analysis completed! {len(saved_plots)} outputs saved to: {output_dir}")
    
    return saved_plots
