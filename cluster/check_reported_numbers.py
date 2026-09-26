#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the cross-dataset numbers printed in the thesis, the paper and the decks
against the sweep CSVs.

Three rounds of review found the same class of defect three times: an interval
written with ordinary rounding, which moves a minimum up and a maximum down and
so cuts off the very runs it is meant to cover. One of them left a run outside
both regime bands, contradicting the sentence that counted seven runs in one of
them. Every edge here has to be floored (lower) or ceilinged (upper).

`checks` lists what the documents currently claim; `bands` lists every interval
they print. Update both when a number in the text changes, and run this before
committing a change to the cross-dataset section:

    python3 cluster/check_reported_numbers.py
"""
import csv, glob, io, math, re, statistics
R = [r for p in sorted(glob.glob("results/ml_11_cross_dataset/sweep/*.csv"))
       for r in csv.DictReader(open(p))]
def sel(t,k): return [r for r in R if r["train"]==t and r["kind"]==k]
def c(rows,k): return [float(r[k]) for r in rows]
A2B, B2A = sel("CICIDS2017","cross"), sel("CSE-CIC-IDS2018","cross")
inA, inB = sel("CICIDS2017","in-domain"), sel("CSE-CIC-IDS2018","in-domain")
f_ab = c(A2B,"f1")
low  = [r for r in A2B if float(r["f1"]) < 0.12]
high = [r for r in A2B if float(r["f1"]) >= 0.12]
t12 = 2.201
def ci(v): return t12*statistics.stdev(v)/len(v)**0.5
mean_ab, ci_ab = statistics.mean(f_ab), ci(f_ab)
srt = sorted(c(B2A,"f1"))
maxgap = max(round(srt[i+1]-srt[i],4) for i in range(len(srt)-1))

checks = [
 ("số lượt tổng", 12, len(A2B)),
 ("số hạt giống", 7, len({r["seed"] for r in A2B})),
 ("hạt giống lặp 2 lượt", 5, sum(1 for s in {r["seed"] for r in A2B}
                                  if sum(1 for r in A2B if r["seed"]==s)==2)),
 ("n chế độ thấp", 7, len(low)),
 ("n chế độ cao", 5, len(high)),
 ("trung bình A->B", 0.1321, round(mean_ab,4)),
 ("CI dưới", 0.0615, round(mean_ab-ci_ab,4)),
 ("CI trên", 0.2027, round(mean_ab+ci_ab,4)),
 ("lượt trong CI", 2, sum(mean_ab-ci_ab<=x<=mean_ab+ci_ab for x in f_ab)),
 ("khe hở lớn nhất B->A", 0.0036, maxgap),
 ("F1 B->A trung bình", 0.0539, round(statistics.mean(c(B2A,"f1")),4)),
 ("CI B->A", 0.0022, round(ci(c(B2A,"f1")),4)),
 ("AUC-PR A->B (mọi lượt)", 0.4862, round(statistics.mean(c(A2B,"auc_pr")),4)),
 ("AUC-PR chế độ thấp", 0.4837, round(statistics.mean(c(low,"auc_pr")),4)),
 ("AUC-PR chế độ cao", 0.4897, round(statistics.mean(c(high,"auc_pr")),4)),
 ("AUC-PR B->A", 0.2072, round(statistics.mean(c(B2A,"auc_pr")),4)),
 ("F1 cùng bộ A", 0.9952, round(statistics.mean(c(inA,"f1")),4)),
 ("F1 cùng bộ B", 0.9246, round(statistics.mean(c(inB,"f1")),4)),
 ("AUC-PR cùng bộ A", 0.9997, round(statistics.mean(c(inA,"auc_pr")),4)),
 ("AUC-PR cùng bộ B", 0.9676, round(statistics.mean(c(inB,"auc_pr")),4)),
 ("CV AUC-PR A->B %", 4.9, round(100*statistics.stdev(c(A2B,"auc_pr"))/statistics.mean(c(A2B,"auc_pr")),1)),
]
bad = 0
print("%-26s %10s %10s %s" % ("đại lượng in trong bài","in","tính lại","kq"))
for n,a,b in checks:
    ok = (a == b) if isinstance(a,int) else abs(a-b) <= 5e-5 + (0.05 if n.endswith('%') else 0)
    bad += not ok
    print("%-26s %10s %10s %s" % (n, a, b, "OK" if ok else "SAI"))

# bands quoted in the documents must contain their runs
bands = {"bảng LV/FAIR thấp": (c(low,"f1"), 0.029, 0.085),
         "bảng LV/FAIR cao":  (c(high,"f1"), 0.185, 0.301),
         "chú thích thấp":    (c(low,"f1"), 0.0291, 0.0843),
         "chú thích cao":     (c(high,"f1"), 0.1853, 0.3009),
         "recall thấp":       (c(low,"recall"), 0.0148, 0.0447),
         "recall cao":        (c(high,"recall"), 0.1036, 0.1778),
         "AUC-PR thấp":       (c(low,"auc_pr"), 0.4590, 0.5163),
         "AUC-PR cao":        (c(high,"auc_pr"), 0.4532, 0.5357),
         "AUC-PR chú thích":  (c(A2B,"auc_pr"), 0.4532, 0.5357),
         "F1 B->A":           (c(B2A,"f1"), 0.0483, 0.0587)}
print()
for n,(v,lo,hi) in bands.items():
    out = [x for x in v if x < lo or x > hi]; bad += len(out)
    print("  %-20s [%.4f, %.4f] ngoài dải %d/%d" % (n, lo, hi, len(out), len(v)))
gap = (max(c(low,"f1")), min(c(high,"f1")))
print("\n  trung bình %.4f trong khoảng trống (%.4f, %.4f): %s" % (mean_ab,*gap,"ĐÚNG" if gap[0]<mean_ab<gap[1] else "SAI"))
bad += not (gap[0] < mean_ab < gap[1])
print("\n%s" % ("== TẤT CẢ KHỚP ==" if bad==0 else "== CÒN %d CHỖ SAI ==" % bad))
