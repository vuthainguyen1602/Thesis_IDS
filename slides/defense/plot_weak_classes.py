"""Slide-sized view of where the weak attack classes go (defense deck, KQ4).

The thesis figure is the full 15x15 confusion matrix, unreadable on a
projector. This draws the same counts (results/ml_09_multiclass_eval/
confusion_matrix_counts.csv) as one 100% bar per class, split into
correct class / another attack class (still flagged at the binary level) /
predicted Benign (missed). DoS Hulk is shown first as a reference.

Run from the repository root:  python3 slides/defense/plot_weak_classes.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/ml_09_multiclass_eval/confusion_matrix_counts.csv"
OUT = Path(__file__).resolve().parent / "img/weak_classes_vi.png"

# (row label in the CSV, label on the slide)
CLASSES = [
    ("DoS Hulk", "DoS Hulk\n(đối chứng)"),
    ("Web Attack - Brute Force", "Web – Brute Force"),
    ("Bot", "Bot"),
    ("Web Attack - XSS", "Web – XSS"),
    ("Infiltration", "Infiltration"),
    ("Web Attack - Sql Injection", "Web – SQLi"),
]
# Okabe-Ito accents used across the decks; validated as a 3-slot palette.
GREEN, BLUE, RED = "#009E73", "#0072B2", "#C1121F"
INK, MUTED = "#23373B", "#6B7B80"

cm = pd.read_csv(SRC, index_col=0)
rows = []
for key, label in CLASSES:
    r = cm.loc[key]
    n = int(r.sum())
    correct, benign = int(r[key]), int(r["BENIGN"])
    rows.append((label, n, correct / n, (n - correct - benign) / n, benign / n))

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 19})
fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
for i, (label, n, c, o, b) in enumerate(rows):
    y = len(rows) - 1 - i
    left = 0.0
    for frac, color in ((c, GREEN), (o, BLUE), (b, RED)):
        if frac > 0:
            ax.barh(y, frac, left=left, height=0.62, color=color,
                    edgecolor="white", linewidth=1.5)
            if frac >= 0.12:  # direct label only where it fits
                ax.text(left + frac / 2, y, f"{frac:.0%}", ha="center", va="center",
                        color="white", fontsize=18, fontweight="bold")
        left += frac
    ax.text(1.02, y, f"{n:,}".replace(",", "."), va="center", ha="left",
            color=MUTED, fontsize=16)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in reversed(rows)], color=INK)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.tick_params(axis="x", colors=MUTED, labelsize=15)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#C9D1D3")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, BLUE, RED)]
ax.legend(handles, ["Đúng lớp", "Lớp tấn công khác", "Lọt thành Benign"],
          loc="upper center", bbox_to_anchor=(0.38, 1.25), ncol=2, frameon=False,
          fontsize=16, handlelength=1.0, columnspacing=1.0, labelcolor=INK)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
for r in rows:
    print(f"{r[0]:<24} n={r[1]:>6}  correct={r[2]:.3f}  other={r[3]:.3f}  benign={r[4]:.3f}")
print(OUT)
