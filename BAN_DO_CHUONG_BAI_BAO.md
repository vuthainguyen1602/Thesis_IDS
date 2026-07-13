# Bản đồ Chương luận án ↔ Bài báo — "IDS biên đáng tin"

*Kịch bản 1: **IDS biên đáng tin** — thích ứng trôi khái niệm (Trục A) + bền vững đối kháng (Trục C) + khoa học đánh giá loại-trừ-rò-rỉ (Trục F, sợi chỉ đỏ) + hiệu quả phần cứng biên (Trục G). Kế thừa luận văn thạc sĩ: IDS trên Apache Spark, cụm Mac + 2 Jetson Orin Nano, đánh giá loại-trừ-rò-rỉ (leakage-aware), anomaly gate (autoencoder), SHAP, so sánh engine suy luận sklearn/ONNX/Spark.*

> **Lưu ý quan trọng (đọc trước):** Tất cả gợi ý về venue, tier (Q1/Q2, rank A/B/C) và deadline dưới đây là **gợi ý định hướng**, không phải khẳng định. Tier tạp chí (Scimago/JCR) và rank hội nghị (CORE) thay đổi hàng năm; scope và deadline cần **kiểm tra trực tiếp** trên trang chính thức của từng venue tại thời điểm nộp. Không có cam kết nào về khả năng được chấp nhận (acceptance) — đó phụ thuộc vào chất lượng công trình và phản biện.

---

## 1. Sơ đồ cấu trúc luận án (9 chương)

Mỗi trục đóng góp = 1 chương; trục F (đánh giá) vừa là một chương riêng vừa là **sợi chỉ đỏ** xuyên suốt mọi chương khác.

```
┌─────────────────────────────────────────────────────────────────────┐
│  CHƯƠNG 1 — MỞ ĐẦU                                                    │
│  Bối cảnh IoT/biên, vấn đề "IDS đáng tin", câu hỏi nghiên cứu,         │
│  mục tiêu, đóng góp, cấu trúc luận án.                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHƯƠNG 2 — TỔNG QUAN & SYSTEMATIC REVIEW                             │
│  Khảo sát hệ thống: ML/DL-IDS cho biên/IoT; trôi khái niệm;           │
│  bền vững đối kháng; rò rỉ & độ tin cậy đánh giá; hiệu quả phần cứng.  │
│  Xác lập khoảng trống → định vị 4 trục đóng góp.        → BÀI 1 (survey)│
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌──────────────┬──────┴───────┬──────────────┐
        ▼              ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ CHƯƠNG 3   │ │ CHƯƠNG 4   │ │ CHƯƠNG 5   │ │ CHƯƠNG 6   │
│ TRỤC F     │ │ TRỤC A     │ │ TRỤC C     │ │ TRỤC G     │
│ Khoa học   │ │ Thích ứng  │ │ Bền vững   │ │ Hiệu quả   │
│ đánh giá   │ │ trôi khái  │ │ đối kháng  │ │ phần cứng  │
│ loại-trừ-  │ │ niệm /     │ │ /evasion   │ │ biên       │
│ rò-rỉ      │ │ học liên   │ │ trên biên  │ │ (TinyML/   │
│ (NỀN TẢNG) │ │ tục biên   │ │            │ │ TensorRT)  │
│ →BÀI 2,3   │ │ →BÀI 4     │ │ →BÀI 5     │ │ →BÀI 6     │
└────────────┘ └────────────┘ └────────────┘ └────────────┘
        │              │              │              │
        └──────────────┴──────┬───────┴──────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHƯƠNG 7 — TÍCH HỢP HỆ THỐNG (CO-DESIGN)                             │
│  Gộp A+C+F+G thành một hệ IDS biên end-to-end trên cụm Jetson;        │
│  Pareto năng lượng–độ trễ–độ chính xác–robustness;                    │
│  đánh giá tích hợp dưới chuẩn loại-trừ-rò-rỉ + đối thủ thích nghi.     │
│                                                       → BÀI 7 (tạp chí)│
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHƯƠNG 8 — BÀN LUẬN, HẠN CHẾ & ĐE DỌA TÍNH HỢP LỆ (tùy chọn tách)    │
│  (Có thể gộp vào Ch.7 hoặc Ch.9 nếu trường yêu cầu 8 chương)          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CHƯƠNG 9 — KẾT LUẬN & HƯỚNG PHÁT TRIỂN                               │
│  Tổng hợp đóng góp, trả lời câu hỏi nghiên cứu, hướng mở              │
│  (federated – Trục B; zero-day SSL – Trục D; LLM/GNN – Trục E).        │
└─────────────────────────────────────────────────────────────────────┘
```

**Phương án 8 chương:** gộp Chương 8 (Bàn luận) vào Chương 7, giữ 1–8.
**Phương án 9 chương:** tách Bàn luận thành chương riêng (nhiều trường EU/quốc tế ưa kiểu này).

**Vì sao Trục F đặt làm Chương 3 (ngay sau tổng quan):** đây là **nền phương pháp luận** — chuẩn đánh giá loại-trừ-rò-rỉ được dùng lại trong mọi chương A/C/G. Đặt sớm để các chương sau "trích dẫn ngược" về giao thức đánh giá chung, tránh lặp lại và giữ tính nhất quán.

---

## 2. Bảng ánh xạ Chương ↔ Bài báo

| Chương | Nội dung cốt lõi | Đóng góp mới | Bài báo | Loại | Venue gợi ý (tier — cần kiểm tra) | Trạng thái |
|---|---|---|---|---|---|---|
| **Ch.2** Tổng quan | Systematic review: IDS biên đáng tin qua 4 lăng kính (drift, đối kháng, rò rỉ/đánh giá, hiệu quả phần cứng) | Khung phân loại (taxonomy) + bản đồ khoảng trống nối 4 trục; tiêu chí "đáng tin" cho IDS biên | **BÀI 1 — Survey** | Tạp chí (review) | *ACM Computing Surveys* (Q1); *Computer Science Review* (Q1, Elsevier); *IEEE Communications Surveys & Tutorials* (Q1, nếu nghiêng mạng) | **Cần làm** (ưu tiên sớm) |
| **Ch.3** Trục F (nền) | Quy trình loại-trừ-rò-rỉ tự động + kiểm định thống kê; mở rộng cross-dataset; feature selection chống rò rỉ | Công cụ kiểm toán rò rỉ (temporal / near-duplicate / label-leakage) + giao thức đánh giá tái lập | **BÀI 2 — FAIR'2026** *(đã có)* | Hội nghị | **FAIR 2026** (HN quốc gia Việt Nam — đã nộp/đã có) | **Đã có** |
| **Ch.3** Trục F (mở rộng) | Nâng FAIR'2026 thành chuẩn benchmark mở + đối thủ thích nghi + thước đo "sim-to-real gap" | Leaderboard tái lập + bộ kiểm toán rò rỉ tổng quát hóa đa-dataset | **BÀI 3 — Leakage benchmark** | Tạp chí | *Computers & Security* (Q1, Elsevier); *Cybersecurity* (SpringerOpen, Q2); *IEEE Access* (Q1/Q2, nhanh) | **Cần làm** (kế thừa BÀI 2) |
| **Ch.4** Trục A | Học liên tục / online learning dưới trôi khái niệm trên Jetson; replay buffer chọn lọc; phát hiện trôi rẻ chạy ARM | Khung continual-learning **dưới ràng buộc tài nguyên biên** + đánh giá theo `IDS_SPLIT_MODE=temporal` | **BÀI 4 — Drift/continual** | Hội nghị (→ mở rộng tạp chí) | **IEEE LCN**; **IEEE/ACM SEC** (edge); **SOICT** (CCIS); tạp chí: *Future Generation Computer Systems* (Q1) | **Cần làm** |
| **Ch.5** Trục C | Bền vững đối kháng/evasion **khả thi trên biên**; tấn công né cả anomaly gate lẫn classifier; ràng buộc khả thi miền mạng | Phân tích robustness pipeline 2 tầng (gate+classifier) + phòng thủ thời-gian-thực trên Jetson | **BÀI 5 — Adversarial** | Hội nghị an ninh | **RAID**; **ESORICS** (+ workshop SiMLA); **ACSAC**; tạp chí: *IEEE TIFS* (Q1) hoặc *Computers & Security* (Q1) | **Cần làm** |
| **Ch.6** Trục G | Đồng thiết kế mô hình–phần cứng: PTQ/quantization, ONNX→TensorRT, khai thác GPU/Tensor core Jetson; Pareto năng lượng–chính xác | Tối ưu đa mục tiêu năng lượng–độ trễ–chính xác–robustness; lượng tử hóa **không mất robustness** | **BÀI 6 — SOICT** *(đã có, edge deploy)* | Hội nghị | **SOICT** (CCIS, Springer — đã có) → có thể mở rộng tạp chí TinyML/edge | **Đã có** (cần mở rộng cho Trục G) |
| **Ch.7** Tích hợp | Hệ IDS biên end-to-end gộp A+C+F+G; co-design; đánh giá tích hợp dưới chuẩn rò-rỉ + đối thủ | Bằng chứng hệ thống: thích ứng + bền vững + hiệu quả **đồng thời** trên cụm biên thật | **BÀI 7 — System journal** | Tạp chí (tổng kết) | *IEEE Internet of Things Journal* (Q1); *IEEE TIFS* (Q1); *Future Generation Computer Systems* (Q1) | **Cần làm** (cuối, dùng kết quả BÀI 3–6) |

**Tóm tắt số lượng:** 2 bài đã có (BÀI 2 FAIR'2026, BÀI 6 SOICT) + 5 bài cần làm (1 survey, 1 mở rộng F, 1 trục A, 1 trục C, 1 tích hợp tạp chí) = **7 bài**. Có thể rút gọn còn 5–6 nếu tiến độ yêu cầu (xem §5).

---

## 3. Gợi ý venue cụ thể & lý do *(đều cần kiểm tra scope/deadline)*

### Hội nghị

- **FAIR (Nghiên cứu cơ bản & ứng dụng CNTT, Việt Nam)** — đã dùng cho BÀI 2. Phù hợp công bố trong nước, tốc độ nhanh, hợp để "đặt cọc" đóng góp Trục F sớm.
- **SOICT (Int. Symposium on Information and Communication Technology, Việt Nam)** — đã dùng cho BÀI 6. Kỷ yếu Springer **CCIS** (có chỉ mục Scopus), có track Cyber Security & Networking. SOICT 2025 tổ chức tại Nha Trang (12–14/12/2025); **SOICT 2026 cần xem soict.org để chốt deadline**. Tốt cho phần edge deployment / hiệu quả phần cứng.
- **IEEE LCN (Local Computer Networks)** và **IEEE/ACM Symposium on Edge Computing (SEC)** — hợp cho BÀI 4 (continual learning trên biên): nhấn mạnh ràng buộc tài nguyên, mạng thật.
- **RAID (Research in Attacks, Intrusions and Defenses)** — rất hợp BÀI 5: RAID nêu rõ scope gồm intrusion detection, ML for security, **security of ML systems**, và **adversarial learning**. Đây là venue an ninh có uy tín cho phần đối kháng.
- **ESORICS** (+ workshop **SiMLA / Security in ML and its Applications**) và **ACSAC** — lựa chọn thay thế cho BÀI 5 nếu lệch deadline RAID; ACSAC mạnh về hệ thống bảo mật ứng dụng.
  - *Tham khảo deadline an ninh:* tổng hợp tại **sec-deadlines.github.io** (kiểm tra trước mỗi vòng nộp).

### Tạp chí

- **ACM Computing Surveys** / **Computer Science Review** (Elsevier) / **IEEE Communications Surveys & Tutorials** — cho **BÀI 1 (survey)**. CSUR và COSREV là venue review hàng đầu; COMST hợp nếu nghiêng về mạng/IoT. Đều Q1, phản biện kỹ, thời gian dài → **nộp sớm**.
- **Computers & Security** (Elsevier, Q1) — phù hợp BÀI 3 (benchmark rò rỉ) và là phương án cho BÀI 5/BÀI 7; scope an ninh thực nghiệm, đánh giá nghiêm.
- **IEEE Transactions on Information Forensics and Security (TIFS)** (Q1) — venue tạp chí hàng đầu cho an ninh + ML; lựa chọn tham vọng cho BÀI 5 hoặc BÀI 7.
- **IEEE Internet of Things Journal** (Q1) — rất hợp BÀI 7 (hệ thống IDS biên end-to-end trên IoT).
- **Future Generation Computer Systems** (Elsevier, Q1) — hợp BÀI 4 mở rộng hoặc BÀI 7 (edge/distributed systems).
- **IEEE Access** (Q1/Q2) / **Cybersecurity** (SpringerOpen) — phương án "an toàn", phản biện nhanh hơn, hợp để bảo đảm đủ số bài nếu các tạp chí top kéo dài.

> Lý do chọn theo trục: survey → venue review; F (đánh giá/dữ liệu) → tạp chí an ninh thực nghiệm; A (biên/streaming) → venue edge/network; C (đối kháng) → venue an ninh có track ML-security; G (phần cứng) → venue edge/TinyML; tích hợp → tạp chí IoT/forensics tổng kết.

---

## 4. Trình tự thực hiện & phụ thuộc giữa các bài

```
Giai đoạn 1 (nền — năm 1)
  BÀI 2 (FAIR'2026, ĐÃ CÓ) ──┐
                              ├──► BÀI 3 (Leakage benchmark, tạp chí)
  BÀI 6 (SOICT, ĐÃ CÓ) ───────┘        │ (chuẩn đánh giá dùng cho mọi bài sau)
                                       │
  BÀI 1 (Survey) ── nộp song song, sớm ─┘ (định vị khoảng trống, hỗ trợ Ch.2)

Giai đoạn 2 (đóng góp lõi — năm 2)
  BÀI 3 (chuẩn F) ──► BÀI 4 (Trục A — drift)   [dùng giao thức temporal split + leakage-free]
                 └──► BÀI 5 (Trục C — đối kháng) [dùng chuẩn đánh giá + đối thủ thích nghi]
  BÀI 6 (đã có) ───► mở rộng Trục G (quantization/TensorRT, robustness-aware)

Giai đoạn 3 (tích hợp — năm 3)
  BÀI 4 + BÀI 5 + BÀI 6(mở rộng) + BÀI 3 ──► BÀI 7 (System journal, tổng kết)
                                            └──► hoàn thiện Ch.7 + viết luận án
```

**Logic phụ thuộc:**
1. **BÀI 3 là then chốt** — nó cố định *giao thức đánh giá loại-trừ-rò-rỉ* mà BÀI 4, 5, 7 đều phải tuân theo. Làm sớm để tránh phải đánh giá lại các bài sau.
2. **BÀI 1 (survey) làm sớm** vì nó vừa nuôi Chương 2 vừa giúp định vị chính xác khoảng trống cho 3 bài đóng góp → tăng tính thuyết phục khi phản biện.
3. **BÀI 2 và BÀI 6 (đã có)** là điểm tựa: BÀI 2 → mở rộng thành BÀI 3; BÀI 6 → mở rộng thành Trục G/Chương 6.
4. **BÀI 4 và BÀI 5 độc lập với nhau** → có thể chạy song song khi đã có chuẩn F.
5. **BÀI 7 làm cuối cùng** vì nó gộp kết quả của tất cả các bài trước thành bằng chứng hệ thống tích hợp.

---

## 5. Tiêu chí "đủ điều kiện bảo vệ" (điển hình — **phải kiểm tra quy định trường**)

> Đây là **mức tham khảo chung** ở Việt Nam và quốc tế; quy định cụ thể do **cơ sở đào tạo / quy chế NCS** quyết định và thay đổi theo thời gian. Bắt buộc tra cứu quy chế hiện hành của trường bạn.

- **Khung phổ biến ở Việt Nam (tham khảo):** NCS thường cần một số tối thiểu công bố là **tác giả chính/đứng đầu**, trong đó **ít nhất 1–2 bài trên tạp chí có chỉ mục quốc tế (Scopus/WoS)**, phần còn lại có thể là hội nghị/tạp chí trong danh mục được tính điểm (HĐGSNN). Một số trường yêu cầu **≥1 bài Q1/Q2** hoặc tổng điểm công trình đạt ngưỡng.
- **Vai trò tác giả:** nhiều quy chế chỉ tính bài mà NCS là **tác giả đầu / tác giả liên hệ**, và có thể yêu cầu **người hướng dẫn là đồng tác giả**. Kiểm tra kỹ.
- **Khuyến nghị an toàn cho kế hoạch này:** nhắm tới **ít nhất 2 bài tạp chí Scopus/WoS (ưu tiên ≥1 bài Q1)** + **2–3 bài hội nghị có kỷ yếu chỉ mục** (FAIR/SOICT/CCIS, hoặc hội nghị an ninh quốc tế). Bộ 7 bài đề xuất ở trên **vượt mức tối thiểu**, tạo dư địa an toàn nếu một số bài bị từ chối/kéo dài.
- **Phương án rút gọn (nếu thời gian gấp):** giữ **BÀI 1 (survey), BÀI 3 (tạp chí F), BÀI 7 (tạp chí tích hợp)** làm 3 trụ tạp chí; BÀI 4/5 ở dạng hội nghị; bỏ qua phần mở rộng nếu cần. Tối thiểu khả thi: **survey + 1 tạp chí Q1 + 2–3 hội nghị**.

### 5.1 — Áp cho ĐH Công nghiệp TP.HCM (IUH): mục tiêu **1 Q1 + 2 Q3 + 2 trong nước**

> ⚠️ **Cần xác minh:** Thông báo tuyển sinh TS công khai của IUH (ipe.iuh.edu.vn) chỉ nêu điều kiện **đầu vào**. Chuẩn công bố để **bảo vệ (đầu ra)** nằm trong *Quy chế đào tạo tiến sĩ nội bộ của IUH* — phải lấy trực tiếp từ **Viện Đào tạo Quốc tế & Sau đại học IUH** / phòng SĐH ngành CNTT–ATTT. Mọi nhận định dưới đây là mức chung.

- **Sàn quốc gia (Thông tư 18/2021):** NCS là **tác giả chính**, tổng **≥ 2,0 điểm**, mix linh hoạt — tạp chí WoS/Scopus, hội nghị quốc tế uy tín, **hoặc tạp chí trong nước HĐGSNN ≥ 0,75đ**.
- **Đánh giá mục tiêu "1 Q1 + 2 Q3 + 2 trong nước":** nếu Q1/Q3 là **tạp chí WoS/Scopus** và bạn là **tác giả chính** → **vượt yêu cầu thoải mái** (3 bài Q-ranked đã hơn mọi mức "1–2 Scopus journal" + dư điểm sàn 2,0). Còn **biên an toàn**: trượt 1 bài vẫn đủ.

| Hạng mục mục tiêu | Bài trong kế hoạch | Ghi chú |
|---|---|---|
| **Q1** (×1) | **BÀI 1 — Survey** *(ưu tiên)* hoặc BÀI 3 (benchmark F) / BÀI 7 (tích hợp) | Survey ở tạp chí review (CSUR, Computer Science Review) là **đường vào Q1 khả thi nhất** cho NCS |
| **Q3** (×2) | **BÀI 4 (trôi khái niệm)** + **BÀI 5 (đối kháng)** ở tạp chí Q2/Q3 | Hoặc mở rộng từ bản hội nghị lên tạp chí (IEEE Access, Cybersecurity, …) |
| **Trong nước** (×2) | **BÀI 2 — FAIR'2026** (HN quốc gia) + **1 tạp chí trong nước HĐGSNN** | FAIR tính là 1 "trong nước"; cần thêm 1 tạp chí trong nước |

- **Lưu ý phân loại:** **SOICT (BÀI 6) là hội nghị *quốc tế*** (kỷ yếu Springer CCIS/Scopus) — **KHÔNG** tính "trong nước"; nó là output quốc tế **cộng thêm** (tốt cho điểm, không nằm trong 2 bài trong nước).
- **Cảnh báo quy chế:** một số trường **chỉ tính tạp chí WoS/Scopus** (không nhận hội nghị thay tạp chí), có thể quy định **Q-rank tối thiểu** và yêu cầu bài **đăng sau khi đã là NCS**, cùng **vai trò tác giả đầu/liên hệ**. Bắt buộc đối chiếu quy chế IUH ngành của bạn trước khi chốt.

---

## Phụ lục — Liên kết kế thừa luận văn → trục NCS

| Tài sản luận văn thạc sĩ | Trục NCS kế thừa | Bài báo |
|---|---|---|
| Quy trình loại-trừ-rò-rỉ + kiểm định thống kê + cross-dataset | Trục F (Ch.3) | BÀI 2, 3 |
| Pipeline streaming Kafka + cụm Jetson (`IDS_SPLIT_MODE=temporal`) | Trục A (Ch.4) | BÀI 4 |
| Anomaly gate (autoencoder) + ensemble | Trục C (Ch.5) | BÀI 5 |
| Baseline engine Spark/sklearn/ONNX + đo năng lượng tegrastats | Trục G (Ch.6) | BÀI 6 |
| Toàn bộ hạ tầng + SHAP + cụm biên thật | Tích hợp (Ch.7) | BÀI 7 |

*Các trục B (federated), D (zero-day/SSL), E (LLM/GNN/XAI) trong HUONG_PHAT_TRIEN_NCS.md được giữ làm **hướng phát triển tương lai** ở Chương 9, không nằm trong phạm vi 4 trục lõi của Kịch bản 1.*

---

*Nguồn tham khảo định hướng venue (kiểm chứng 2025–2026):*
- *RAID 2026 — scope gồm intrusion detection, ML for security, adversarial learning: myhuiban.com/conference/64*
- *SOICT — soict.org; kỷ yếu Springer CCIS: link.springer.com/conference/soict*
- *Tổng hợp deadline hội nghị an ninh: sec-deadlines.github.io*
- *ESORICS workshop SiMLA (Security in ML and its Applications) co-located 2026*

> **Nhắc lại:** mọi tier/rank/deadline ở trên là gợi ý, không khẳng định; hãy xác minh trên trang chính thức và đối chiếu quy chế NCS của trường trước khi quyết định.
