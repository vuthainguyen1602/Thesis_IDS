#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 10 — Feature-leakage ablation (destination_port).

Quantifies how much the label-leaking ``destination_port`` feature inflates
binary IDS performance. Every classifier is trained TWICE on the *same*
train/test split:

  (A) WITHOUT port features  -> the leak-free, reported configuration
  (B) WITH    port features  -> the leaky configuration

Both feature sets come from the identical parquet split (the port columns are
retained in the parquet by ml_00; only feature selection differs), so the only
varying factor is the presence of the leaky feature.

The ablation runs across the same eight classifiers as the rest of the study —
a FAIR'2026 reviewer asked for exactly this, since a port effect measured on
one model says little about the others. Hyper-parameters come from
``get_classifiers()``, the factory ml_01/ml_07 use, so the rows stay comparable
with the main tables instead of drifting from tab:hparams.

Cost: two fits per model. On the 2-Jetson cluster the full sweep is ~4 h,
dominated by XGBoost and GBT. ``IDS_ABLATION_MODELS`` restricts the sweep
(e.g. ``IDS_ABLATION_MODELS="Random Forest,Decision Tree"``) and the per-model
CSV is rewritten after every model, so an interrupted run keeps what it had.

Outputs (results/ml_10_leakage_ablation/):
  - leakage_ablation.csv            Random Forest only, original schema —
                                    feeds FAIR'2026 tab:port-ablation and the
                                    thesis table, unchanged
  - leakage_ablation.png            grouped bar chart for that model
  - leakage_ablation_all_models.csv one row per model x arm, plus delta_f1
  - leakage_ablation_by_model.png   delta F1 (leaky - leak-free) per model

This script is additive: it does not modify any other experiment.
"""
import os
import pandas as pd

from shared_utils import (
    create_spark_session,
    load_and_prepare_data,
    Pipeline,
    VectorAssembler,
    StandardScaler,
)
from idslib.data import _leaky_port_cols
from idslib.modeling import train_and_evaluate, get_classifiers, add_class_weights

OUT_DIR = os.path.join("results", "ml_10_leakage_ablation")
os.makedirs(OUT_DIR, exist_ok=True)

# The model whose two rows stay in the legacy CSV/plot the manuscripts cite.
HEADLINE_MODEL = os.environ.get("IDS_ABLATION_HEADLINE", "Random Forest")
# Subset of the classifier set to sweep; "all" keeps the full eight.
MODELS_FILTER = os.environ.get("IDS_ABLATION_MODELS", "all").strip()

METRIC_KEYS = ("f1", "recall", "precision", "auc_pr")
NO_PORT_LABEL = "Without destination_port (proposed)"
WITH_PORT_LABEL = "With destination_port (leaky)"


def _build_pipeline(feature_cols, clf):
    """Assemble -> scale -> classify, matching the main experiments' stages."""
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features_scaled",
        withMean=True, withStd=True,
    )
    return Pipeline(stages=[assembler, scaler, clf])


def _classifiers_for(feature_cols, scale_pos_weight):
    """Classifier set for one arm. num_features differs between the arms (the
    leaky arm has the port columns), which only affects the MLP layer sizes."""
    classifiers = get_classifiers(
        features_col="features_scaled",
        num_features=len(feature_cols),
        scale_pos_weight=scale_pos_weight,
        seed=42,
    )
    if MODELS_FILTER.lower() != "all":
        wanted = [m.strip() for m in MODELS_FILTER.split(",") if m.strip()]
        missing = [m for m in wanted if m not in classifiers]
        if missing:
            print(f"  [WARN] Unknown model(s) in IDS_ABLATION_MODELS: {missing}")
        classifiers = type(classifiers)(
            (k, v) for k, v in classifiers.items() if k in wanted)
    return classifiers


def _write_all_models_csv(rows):
    """Rewritten after every model so an interrupted sweep is still usable."""
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "leakage_ablation_all_models.csv")
    df.to_csv(path, index=False)
    return df, path


def _write_headline_csv(rows):
    """Legacy artifact: the headline model's two rows in the original schema,
    so the manuscripts' table and figure keep reproducing byte-for-byte."""
    head = [r for r in rows if r["model"] == HEADLINE_MODEL]
    if not head:
        return None
    df = pd.DataFrame([
        {"config": r["config"], "port_included": r["port_included"],
         **{k: r.get(k) for k in METRIC_KEYS}}
        for r in sorted(head, key=lambda r: r["port_included"])
    ])
    path = os.path.join(OUT_DIR, "leakage_ablation.csv")
    df.to_csv(path, index=False)
    print(f"\n[INFO] Saved: {path}  ({HEADLINE_MODEL}, schema unchanged)")
    print(df.to_string(index=False))
    return df


def _plot_headline(df):
    """Grouped bar chart for the headline model (FAIR figure)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["F1", "Recall", "Precision", "AUC-PR"]
        x = np.arange(len(METRIC_KEYS))
        width = 0.38

        fig, ax = plt.subplots(figsize=(8, 5))
        no_vals = [df.loc[~df.port_included, k].values[0] if (~df.port_included).any()
                   else 0 for k in METRIC_KEYS]
        ax.bar(x - width / 2, [v if v is not None else 0 for v in no_vals], width,
               label="Without port (proposed)", color="#2ca02c")
        if df.port_included.any():
            with_vals = [df.loc[df.port_included, k].values[0] for k in METRIC_KEYS]
            ax.bar(x + width / 2, [v if v is not None else 0 for v in with_vals], width,
                   label="With port (leaky)", color="#d62728")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score (binary)")
        ax.set_title(f"Feature-leakage ablation: effect of destination_port "
                     f"({HEADLINE_MODEL})")
        ax.legend()
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
        fig.tight_layout()
        png_path = os.path.join(OUT_DIR, "leakage_ablation.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved: {png_path}")
    except Exception as e:
        print(f"[WARN] Ablation plot skipped: {e}")


def _plot_by_model(pairs):
    """delta F1 (leaky - leak-free) per model: where the port actually helps."""
    if not pairs:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        ordered = sorted(pairs, key=lambda p: p["delta_f1"])
        names = [p["model"] for p in ordered]
        deltas = [p["delta_f1"] for p in ordered]
        colors = ["#d62728" if d > 0 else "#2ca02c" for d in deltas]

        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(names))
        ax.barh(y, deltas, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("F1(with port) − F1(without port)")
        ax.set_title("How much destination_port inflates binary F1, per model")
        for i, d in enumerate(deltas):
            ax.text(d, i, f" {d:+.4f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8)
        fig.tight_layout()
        png_path = os.path.join(OUT_DIR, "leakage_ablation_by_model.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved: {png_path}")
    except Exception as e:
        print(f"[WARN] Per-model plot skipped: {e}")


def main():
    spark = create_spark_session("IDS_Exp10_Leakage_Ablation")
    # Loaded with port EXCLUDED (default). We re-add the port columns manually
    # for the leaky run so both runs share the exact same split.
    _df, train_df, test_df, feat_noport = load_and_prepare_data(spark)

    present_ports = [c for c in _leaky_port_cols() if c in train_df.columns]
    feat_withport = feat_noport + present_ports

    # One weighted training frame for every fit: same uniform imbalance handling
    # the paper declares (weightCol for the six weight-aware models; XGBoost
    # gets scale_pos_weight; MLP stays unweighted — a Spark ML limitation noted
    # in the paper's threats to validity).
    #
    # Deliberately NOT cached. The weight column is one literal-valued
    # withColumn, so recomputing it per fit is nearly free, while caching 1.8M
    # rows pins memory on the 8 GB board that also hosts the driver — a measured
    # RF fit went from ~10 min to >25 min with 4 GB of swap in use.
    train_w = add_class_weights(train_df)
    counts = {r["label_binary"]: r["count"]
              for r in train_df.groupBy("label_binary").count().collect()}
    spw = (float(counts.get(0, 0)) / float(counts.get(1, 1))) if counts.get(1) else 1.0

    arms = [(NO_PORT_LABEL, False, feat_noport)]
    if present_ports:
        arms.append((WITH_PORT_LABEL, True, feat_withport))

    model_names = list(_classifiers_for(feat_noport, spw).keys())

    print("\n" + "=" * 70)
    print("  EXPERIMENT 10: FEATURE-LEAKAGE ABLATION (destination_port)")
    print("=" * 70)
    print(f"  Features without port: {len(feat_noport)}")
    print(f"  Port columns re-added : {present_ports if present_ports else 'NONE FOUND'}")
    print(f"  Features with port    : {len(feat_withport)}")
    print(f"  Models ({len(model_names)}): {', '.join(model_names)}")
    print(f"  Fits to run          : {len(model_names) * len(arms)}")

    if not present_ports:
        print("  [WARN] No port columns present in the parquet — cannot run the "
              "leaky arm. Re-run ml_00 without dropping port columns.")

    rows = []
    for name in model_names:
        for config, port_included, feature_cols in arms:
            clf = _classifiers_for(feature_cols, spw)[name]
            title = f"{name} — {'with' if port_included else 'without'} destination_port"
            _model, _preds, metrics = train_and_evaluate(
                _build_pipeline(feature_cols, clf), train_w, test_df, title)
            rows.append({
                "model": name,
                "config": config,
                "port_included": port_included,
                "n_features": len(feature_cols),
                **{k: metrics.get(k) for k in METRIC_KEYS},
            })
        _write_all_models_csv(rows)  # checkpoint after each model

    df_all, all_path = _write_all_models_csv(rows)
    print(f"\n[INFO] Saved: {all_path}")
    print(df_all.to_string(index=False))

    # ── Per-model port effect ────────────────────────────────────────────────
    pairs = []
    for name in model_names:
        with_row = next((r for r in rows if r["model"] == name and r["port_included"]), None)
        no_row = next((r for r in rows if r["model"] == name and not r["port_included"]), None)
        if with_row and no_row and with_row.get("f1") is not None and no_row.get("f1") is not None:
            pairs.append({"model": name,
                          "f1_without_port": no_row["f1"],
                          "f1_with_port": with_row["f1"],
                          "delta_f1": with_row["f1"] - no_row["f1"]})
    if pairs:
        delta_df = pd.DataFrame(sorted(pairs, key=lambda p: -p["delta_f1"]))
        delta_path = os.path.join(OUT_DIR, "leakage_ablation_delta_f1.csv")
        delta_df.to_csv(delta_path, index=False)
        print(f"\n[INFO] Port effect per model (F1 with − without) → {delta_path}")
        print(delta_df.to_string(index=False))

    head_df = _write_headline_csv(rows)
    if head_df is not None:
        _plot_headline(head_df)
    _plot_by_model(pairs)

    spark.stop()


if __name__ == "__main__":
    main()
