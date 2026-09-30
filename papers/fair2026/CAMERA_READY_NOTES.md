# Camera-ready notes — FAIR'2026

What changed between the reviewed submission and the camera-ready, and why. Meant
as the basis for a short cover note to the chair; the wording to paste is at the
bottom.

Baseline for every diff below: `7a2a992`, the state of
`papers/fair2026/manuscript/main_en.tex` when the reviews arrived.

## Changes the reviewers asked for

| Review point | What the camera-ready does |
|---|---|
| "Low novelty" / equivalence claim rests on a non-significant $p$ | TOST equivalence test added (`idslib/modeling.py: tost_equivalence`), reported in Section IV; a non-significant $p$ is no longer the basis for the equivalence claim |
| A port effect measured on one model says little about the others | The `destination_port` ablation now runs all eight classifiers; `tab:port-ablation` carries one row per model |
| "At least one simple domain-adaptation baseline" | Two unsupervised baselines, per-domain scaler refit and CORAL, both without target labels and without retraining; reported with the finding that both drive F1 to zero while CORAL lifts AUC-PR for 2018→2017 |
| — (added on the same theme) | A supervised comparison on a 1% target-label budget, and the observation that pooling the source data is worse than dropping it |
| Ranking fixed across resamplings may leak | The re-ranking check that had been run is now reported: F1 moved by at most $7.4\times10^{-5}$ across two of the six resamplings, with 25–26 of 30 features reselected |
| Deduplication is a second leakage source, unmeasured | Recorded as future work in the conclusion, with the port ablation as the template |

## One change the reviewers did not ask for

The cross-dataset table is not the table they read, and this is the item worth
flagging explicitly.

The reviewed version reported a single cross-dataset F1 per direction: 0.0423 for
CICIDS2017 → CSE-CIC-IDS2018 and 0.0535 for the reverse. Repeating the experiment
showed that quantity is **not reproducible across runs of identical code and
data**, and not in a small way. Over 18 runs with 13 seeds, F1 in the
2017→2018 direction is bimodal: 11 runs land at 0.0291–0.0843 and 7 at
0.1853–0.3009, with nothing in between. Six seeds never used before behaved the
same way. The gap spans 37% of the observed range, so a continuous distribution
would be expected to put about seven of eighteen runs inside it.

The mechanism is in the metric, not the model: AUC-PR is statistically the same in
both regimes (0.4778 against 0.4960), so the ranking the model learns does not
change between runs. What moves is where the source-fitted 0.5 threshold falls
against a cluster of near-identical target attack flows that crosses it as a
block, taking recall from about 0.03 to about 0.15 with no intermediate values.

The camera-ready therefore reports that direction as its two regimes rather than
as a single number or a mean — a mean falls in the empty gap and its 95% interval
holds 1 of the 18 runs, so 0.0423 sat outside an interval meant to describe the
same quantity — and leads the comparison with AUC-PR. The paper's conclusion is
unchanged and, if anything, better supported: near-perfect in-domain scores do not
survive a change of testbed, since even the best of eighteen runs reaches F1 0.30
against an in-domain 0.9953.

Nothing was removed to make room that affects a claim: the added text is paid for
by tightening prose that repeated material stated elsewhere. The paper is still
within 8 pages.

## Reproducibility

- 18 run CSVs, their trimmed logs and the excluded run are in
  `results/ml_11_cross_dataset/sweep/`
- `python3 cluster/xd_sweep_summary.py` reproduces every aggregate
- `python3 cluster/check_reported_numbers.py` verifies each number and interval
  printed in the paper against those CSVs

## Draft cover note

> Dear Chair,
>
> Please find our camera-ready for paper #1571327230. It implements the changes both
> reviewers asked for: a TOST equivalence test in place of relying on a
> non-significant p-value, the destination-port ablation extended from one model
> to all eight, two unsupervised domain-adaptation baselines (per-domain scaler
> refit and CORAL) plus a supervised 1%-target-label comparison, and the
> re-ranking check reported rather than described as future work. The
> deduplication ablation is recorded as future work.
>
> We also want to flag one change the reviewers did not request. The submitted
> version reported a single cross-dataset F1 per direction. Repeating that
> experiment showed the quantity is not reproducible across runs of identical
> code and data: over 18 runs it is bimodal, with 11 runs at 0.029–0.085 and 7 at
> 0.185–0.301 and none in between, while AUC-PR is unchanged between the two
> groups. The camera-ready therefore reports that direction as its two regimes
> and leads the comparison with AUC-PR, instead of quoting one draw. The paper's
> conclusion is unchanged. We are happy to provide the run data if useful.
>
> Two smaller editorial changes: the abstract now states the main findings
> instead of deferring them to the body, and several entries of the
> related-work table were corrected after checking each against the cited
> paper, with one cross-dataset study (Cantone et al., already cited in the
> text) added as a row. We also narrowed the multiclass claims to what was
> measured: the per-class results come from one configuration (Random Forest,
> all features), so the paper no longer says they show where the reduction
> methods differ.
>
> Kind regards,
> Thai Nguyen Vu, Bui Van Dung, Tri Nhut Do, Van Du Nguyen
