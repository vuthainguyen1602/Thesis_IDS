"""F1 heatmap for the defense deck (backup Q5), restricted to what the thesis reports.

Same data as results/ml_07_cross_method_comparison/f1_heatmap.png
(cross_method_summary.csv), but a cell keeps its colour and value only if
the thesis prints that F1; every other cell is greyed out with no value, so
a committee member can find each number shown in the thesis.

Thesis sources of the kept cells:
  Baseline, all 10 models ............ Bảng 4.2 (tab:exp0_ml_metrics)
  RF Top-30 / SHAP Top-30, DT RF GBT XGB  Bảng 4.4 (tab:rf_vs_shap)
  SHAP Top-30 XGB, PCA XGB ........... Bảng 4.3 (tab:summary_comparison)
  RF Top-30 HB 0.9938, PCA HB 0.9940,
  SHAP HB 0.9966, SHAP Soft Voting 0.9967  discussion (Mục 4.4) and RQ2
Not kept although the value appears in the thesis by coincidence:
  PCA x RF 0.9837 (thesis number is RF on the SHAP-RF list),
  RF Top-30 x Soft Voting 0.9922 (thesis number is XGBoost on RF Top-30).

Run from the repository root:  python3 slides/defense/plot_f1_heatmap.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/ml_07_cross_method_comparison/cross_method_summary.csv"
OUT = Path(__file__).resolve().parent / "img/f1_heatmap_vi.png"

METHODS = [("Baseline (All Features)", "Baseline (77)"), ("RF Top-30", "RF Top-30"),
           ("SHAP Top-30", "SHAP Top-30"), ("PCA k=40", "PCA k=40")]
MODELS = [("Decision Tree", "DT"), ("Logistic Regression", "LR"),
          ("SVM (LinearSVC)", "SVM"), ("Naive Bayes", "NB"),
          ("Random Forest", "RF"), ("GBT", "GBT"), ("XGBoost", "XGBoost"),
          ("MLP", "MLP"),
          ("Hybrid Bagging Ensemble (Top-3 Mixed + Weighted)", "Hybrid\nBagging"),
          ("Ensemble Voting", "Soft\nVoting")]
TREES = ["Decision Tree", "Random Forest", "GBT", "XGBoost"]
HB, SV = MODELS[8][0], MODELS[9][0]
IN_THESIS = ({("Baseline (All Features)", m) for m, _ in MODELS}
             | {(meth, m) for meth in ("RF Top-30", "SHAP Top-30") for m in TREES}
             | {("PCA k=40", "XGBoost"), ("RF Top-30", HB), ("PCA k=40", HB),
                ("SHAP Top-30", HB), ("SHAP Top-30", SV)})

f1 = {(r["Method"], r["Model"]): float(r["f1"]) for r in csv.DictReader(open(SRC))}

# One-hue sequential ramp (light -> dark green), as in the original figure.
cmap = LinearSegmentedColormap.from_list("f1", ["#EAF6EC", "#7CC48A", "#1B7A3E", "#0B4A24"])
lo, hi = 0.25, 1.0
INK, MUTED, GREY = "#23373B", "#6B7B80", "#E3E6E8"

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 15})
fig, ax = plt.subplots(figsize=(12.5, 4.6), dpi=200)
for i, (meth, mlab) in enumerate(METHODS):
    for j, (mod, _) in enumerate(MODELS):
        v = f1[(meth, mod)]
        if (meth, mod) in IN_THESIS:
            t = (v - lo) / (hi - lo)
            ax.add_patch(Rectangle((j, i), 1, 1, color=cmap(t), ec="white", lw=2.5))
            ax.text(j + .5, i + .5, f"{v:.4f}".replace(".", ","), ha="center", va="center",
                    fontsize=14.5, color="white" if t > 0.55 else INK)
        else:
            ax.add_patch(Rectangle((j, i), 1, 1, color=GREY, ec="white", lw=2.5))
ax.set_xlim(0, len(MODELS)); ax.set_ylim(len(METHODS), 0)
ax.set_xticks([j + .5 for j in range(len(MODELS))])
ax.set_xticklabels([l for _, l in MODELS], color=INK)
ax.set_yticks([i + .5 for i in range(len(METHODS))])
ax.set_yticklabels([l for _, l in METHODS], color=INK)
ax.tick_params(length=0)
ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
ax.add_patch(Rectangle((0, 4.32), 0.32, 0.32, color=GREY, clip_on=False))
ax.text(0.42, 4.48, "luận văn không báo cáo F1 của ô này", va="center", fontsize=13,
        color=MUTED, clip_on=False)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"kept {len(IN_THESIS)} cells, greyed {len(METHODS) * len(MODELS) - len(IN_THESIS)}")
print(OUT)
