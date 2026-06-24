# IDS Research Briefing — Tuesday, June 23, 2026

Honest dating note: genuinely fresh (≤2 weeks) IDS items this cycle are thin. Two arXiv preprints fall squarely in the window; the rest are recent-but-older anchors flagged with dates so you can judge relevance.

- **Categorical Robustness Assessment for ML-based NIDS** (arXiv:2606.12075, ~mid-June 2026). Proposes an evaluation methodology for how ML-NIDS hold up under perturbations to *categorical* flow features (ports, protocols, flags) rather than the usual continuous-feature adversarial framing — relevant if your thesis touches robustness/eval rigor. https://arxiv.org/abs/2606.12075

- **FlowGuard: Flow Matching for Identity-Independent Detection of Data-Free Model-Stealing on Energy-System IDS** (arXiv:2606.03430, early June 2026; ACM e-Energy / Sustainability Week, Jun 22–25). Uses flow matching as an OOD detector to catch data-free model-extraction queries against a deployed IDS, without relying on client identity — a "securing the IDS model itself" angle rather than detection accuracy. https://arxiv.org/abs/2606.03430

- **SoK: Reshaping Research on Network Intrusion Detection Systems** (arXiv:2604.17556, April 2026 preprint; presenting at ASIA CCS '26, June). Systematization arguing much NIDS ML research rests on flawed evaluation/datasets and proposing corrected methodology — worth citing for a thesis lit-review framing even though the preprint predates the 2-week window. https://arxiv.org/abs/2604.17556

- **BigFlow-NIDS dataset** (Data in Brief, doi:10.1016/j.dib.2026.112530, Feb 2026). 66.9M NetFlow records, 55 attributes, 32 attack classes, merged from NF-UNSW-NB15-v3 / NF-ToN-IoT-v3 / NF-BoT-IoT-v3 / NF-CSE-CIC-IDS2018-v3, shipped in Parquet for big-data/streaming experiments. Useful benchmark if you need scale beyond CIC-IDS2017/18. https://www.sciencedirect.com/science/article/pii/S2352340926000831

- **Anomaly-based intrusion detection on benchmark datasets: a comprehensive evaluation** (Scientific Reports, s41598-026-38317-w, 2026). Cross-dataset evaluation of anomaly NIDS — handy as a comparative baseline reference, though not a method novelty. https://www.nature.com/articles/s41598-026-38317-w

Bottom line: a quiet two weeks. The robustness-evaluation thread (FlowGuard + the categorical-robustness preprint + the SoK) is the most thesis-relevant signal right now; consider tracking arXiv cs.CR `pastweek` directly for the next cycle.
