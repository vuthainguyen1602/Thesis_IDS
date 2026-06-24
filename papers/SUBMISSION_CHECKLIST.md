# Checklist nộp bài — SOICT 2026 (Scopus) + FAIR'2026

> Mục tiêu: 1 bài Scopus (**SOICT 2026** — bài hệ thống/edge) + 1 bài **FAIR'2026** (bài phương pháp/đánh giá).
> Trạng thái hiện tại: khung & nội dung xong; **mọi số liệu còn là placeholder `\ph{}`** → chưa nộp được cho tới khi chạy thực nghiệm và điền số thật.

---

## 0. Chạy thực nghiệm trước (khâu chặn lớn nhất)

Thứ tự chạy trên cụm 2-Jetson (master Mac đã `start_master_mac.sh`, 2 worker `start_worker.sh`):

- [ ] `ml_00_prepare_cicids2017.py` → sinh Parquet + thống kê dữ liệu (n_train, n_test, n_features)
- [ ] (nếu claim 2 bộ dữ liệu) chạy lại `ml_00` với `IDS_DATASET=cicids2018`
- [ ] `ml_07_cross_method_comparison.py` → cross-method F1 (+ `f1_heatmap.png`), robustness, kiểm định thống kê
- [ ] `ml_08_anomaly_gate_autoencoder.py` → `gate_operating_points.csv` + `gate_operating_points.png`
- [ ] `ml_09_multiclass_eval.py` → per-class metrics + `confusion_matrix.png`
- [ ] `ml_10_leakage_ablation.py` → `leakage_ablation.csv` + `leakage_ablation.png` (có/không `destination_port`)
- [ ] (cải tiến) `ml_11_cross_dataset_eval.py` → `cross_dataset_results.csv` + `.png` — cần prep CẢ HAI bộ (CICIDS2017 + CSE-CIC-IDS2018) → điền `tab:cross-dataset` (FAIR)
- [ ] (cải tiến) Năng lượng Jetson: chạy benchmark trên Jetson thật → `avg_power_w`, `energy_per_inference_mj` (tự đo qua `tegrastats`) → điền cột Energy/inf của `tab:benchmark` (SOICT)
- [ ] Đo edge: 3 chế độ → `papers/soict2026/run_benchmarks.sh` (local|run|merge) → `summary.csv`
- [ ] `python papers/soict2026/plot_edge_modes.py` → `edge_modes.png` (throughput + p95 latency theo mode)
- [ ] Ghi version: `java -version`, Spark version, Kafka version (điền `\ph{java}`, `\ph{spark}`, `\ph{kafka}`)

Kết quả đổ về `results/ml_07_cross_method_comparison/`, `results/ml_08_anomaly_gate/`, `results/ml_09_multiclass_eval/`, `results/ml_10_leakage_ablation/`, `papers/soict2026/results/benchmarks/`.

**Hình tự sinh** (đã wire `\IfFileExists` vào bài, không cần dán tay): `f1_heatmap.png`, `confusion_matrix.png`, `leakage_ablation.png`, `train_predict_time.png` (FAIR); `gate_operating_points.png`, `edge_modes.png` (SOICT). Chạy xong các script trên là hình tự hiện khi compile.

---

## 1. FAIR'2026 — `papers/fair2026/manuscript/main.tex`

### 1a. Môi trường / dữ liệu
| Ô `\ph{}` | Nguồn |
|---|---|
| `n_train`, `n_test`, `n_features` | ml_00 (thống kê split) |
| `hardware` | Jetson Orin Nano Super (đã có specs sẵn) |
| `java`, `spark` | `java -version`, Spark version |

### 1b. tab:hparams (siêu tham số)
- [ ] Xác nhận bảng siêu tham số khớp `get_classifiers()` trong `idslib/modeling.py` (cố định a priori)

### 1c. tab:port-ablation + fig:leakage-ablation — nguồn: ml_10
| Ô | Ý nghĩa |
|---|---|
| `f1_with` / `f1_no` | F1 khi GIỮ vs BỎ `destination_port` |
| `rec_with` / `rec_no` | Recall tương ứng |
| `pr_with` / `pr_no` | Precision tương ứng |
| `giữ nguyên/thay đổi` | kết luận định tính |

### 1d. tab:best-results (so sánh 4 phương pháp giảm chiều) — nguồn: ml_07
- [ ] `m1..m4` (tên phương pháp), `methodA/methodB`
- [ ] `f1_1..f1_4`, `pr_1..pr_4`, `s_1..s_4`, `t_1..t_4` (F1/precision/size/time mỗi phương pháp)
- [ ] `recall`, `giảm chiều có cần thiết?` → kết luận

### 1e. tab:stat (kiểm định thống kê) — nguồn: ml_07 (paired permutation, N≥6)
- [ ] `modelA`, `modelB`, `f1A`, `f1B`, `ciA`, `ciB` (CI t-distribution)
- [ ] `pval` + diễn giải `tương đương (p=…) / khác biệt có ý nghĩa`

### 1f. fig:f1-heatmap + fig:cm + tab:perclass — nguồn: collect_results.sh / ml_09
- [ ] `fig:f1-heatmap` ← heatmap F1 (phương pháp × thuật toán)
- [ ] `fig:cm` ← `confusion_matrix.png` (ml_09) → copy vào `papers/fair2026/manuscript/figures/`
- [ ] tab:perclass: `cf1`, `cf2`, `r1`, `r2`, `p1`, `p2`, `sup1`, `sup2` + dòng macro-F1 / weighted-F1

---

## 2. SOICT 2026 — `papers/soict2026/manuscript/main.tex`

### 2a. Môi trường
| Ô | Nguồn |
|---|---|
| `hardware` | Jetson Orin Nano Super (đã có) |
| `spark`, `kafka` | version thực tế |

### 2b. tab:spark-edge-bridge (offline F1 → deployable) — nguồn: ml_07
- [ ] `m1..m4` (mô hình), `nf1..nf4` (số đặc trưng), `f1_1..f1_4` (F1), `s1..s4` (model size)

### 2c. tab:metrics-split (gate operating + edge modes) — nguồn: ml_08 + đo edge
- [ ] `f10/f1A/f1B/f1C` (F1 theo điểm vận hành gate)
- [ ] `lat0/latA/latB/latC` (latency), `p95` (p95 latency)
- [ ] `tp0/tpA/tpB/tpC` (throughput), `gs0/gsA` (gate offload %), `rate`, `load`

### 2d. tab:benchmark (3 chế độ triển khai) — nguồn: đo edge
- [ ] `s1..s4` (số liệu so sánh 3 mode: pipeline split / horizontal scaling / Spark cluster)

### 2e. Hình
- [x] `fig:train-deploy` ← sơ đồ pipeline TikZ (đã vẽ, vector)
- [x] `fig:architecture` ← sơ đồ kiến trúc 2-Jetson TikZ (đã vẽ, vector)
- [ ] `fig:edge-modes` ← `edge_modes.png` (chạy `plot_edge_modes.py` sau benchmark)
- [ ] `fig:gate` ← `gate_operating_points.png` (chạy `ml_08`)

---

## 3. Hoàn thiện hình thức (sau khi điền số)

- [ ] **Thống nhất tác giả & affiliation**: luận văn = Bùi Văn Dũng (UIT) vs bài báo = Nguyễn Vũ Thái (UTC). Chốt tên/đơn vị/email khớp nhau.
- [ ] **Đổi template chính thức**: FAIR template; SOICT/ACM template (hiện cả hai dùng LNCS chỉ để compile thử).
- [ ] **Kiểm tra giới hạn trang** từng nơi.
- [ ] **Rà similarity** (Turnitin/iThenticate) giữa 2 bài + luận văn; viết lại đoạn trùng để tránh tự đạo văn.
- [ ] **Xác minh tình trạng Scopus index** của SOICT kỳ 2026 trước khi chốt là "bài Scopus".
- [ ] Kiểm tra trích dẫn: không còn entry bịa, DOI/URL đúng.
- [ ] Compile sạch bản cuối (xelatex), không còn `\ph{}` đỏ nào.

---

## 4. Kiểm tra nhanh "còn placeholder không"

```bash
grep -rn '\\ph{' papers/fair2026/manuscript/main.tex papers/soict2026/manuscript/main.tex
# → khi nộp được: lệnh này KHÔNG trả về dòng nào.
```
