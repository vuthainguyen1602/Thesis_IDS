# Hướng phát triển nghiên cứu tiến sĩ (NCS) — từ luận văn IDS/Spark/Edge

*Tài liệu phân tích, đối chiếu xu hướng 2024–2026. Mục tiêu: định vị một luận án tiến sĩ kế thừa và nâng tầm luận văn thạc sĩ "Phát hiện xâm nhập dựa trên Apache Spark cho mạng IoT, triển khai biên Jetson, đánh giá loại-trừ-rò-rỉ".*

---

## 1. Luận văn đang đứng ở đâu (điểm tựa cho NCS)

Bốn tài sản của luận văn — cũng là lợi thế hiếm để mở rộng lên tiến sĩ:

1. **Quy trình đánh giá loại-trừ-rò-rỉ + kiểm định thống kê** (loại `destination_port`, khử trùng lặp mức đặc trưng, holdout tách rời, permutation test, đa lớp). Đây là *điểm mạnh khác biệt nhất* — phần lớn công trình IDS vẫn báo F1 ≈ 1.000 do rò rỉ.
2. **Hạ tầng huấn luyện phân tán thật** (cụm Spark Mac + 2 Jetson) + **cầu nối train→deploy** (export PipelineModel ra biên).
3. **Pipeline biên thời gian thực** (Kafka + anomaly gate autoencoder + classifier + đo năng lượng/độ trễ/throughput).
4. **Giải thích mô hình** (SHAP) + **so sánh engine suy luận** (Spark vs sklearn/ONNX).

Một luận án tiến sĩ KHÔNG nên chỉ "thêm mô hình deep learning". Nó cần một **đóng góp phương pháp luận bền vững** giải quyết các *hạn chế cốt lõi* mà chính luận văn đã thừa nhận: (a) chỉ phát hiện tấn công **đã biết** (học có giám sát), (b) chưa **thích ứng trôi khái niệm**, (c) chưa **bền trước đối kháng**, (d) quy mô "Big Data" còn là PoC, (e) đánh giá chỉ trên CICIDS2017.

---

## 2. Bảy trục nghiên cứu mức tiến sĩ

Mỗi trục: *Khoảng trống → Đóng góp NCS → Kế thừa luận văn → Phương pháp → Bài toán con*.

### Trục A — Học liên tục & thích ứng trôi khái niệm (Continual / Online Learning under Concept Drift)
- **Khoảng trống:** Mối đe doạ tiến hoá liên tục; mô hình tĩnh suy giảm theo thời gian. IDS cần thích ứng **không quên thảm khốc** (catastrophic forgetting) các tấn công cũ. Đây là một trong những hướng nóng nhất 2025–2026 (streaming continual learning, memory-replay, knowledge distillation).
- **Đóng góp NCS:** Khung **học liên tục thời gian thực trên biên** kết hợp phát hiện trôi (drift detection) + cập nhật mô hình tăng tiến *dưới ràng buộc tài nguyên Jetson* — điều các công trình trên server bỏ qua.
- **Kế thừa:** pipeline streaming Kafka + cụm Jetson của bạn là *testbed tự nhiên* cho học liên tục trên thiết bị.
- **Phương pháp:** incremental learning, replay buffer có chọn lọc, distillation đa tầng, adaptive normalization; đánh giá theo kịch bản trôi thật (train/test theo thời gian — bạn đã có `IDS_SPLIT_MODE=temporal`).
- **Bài toán con:** (i) phát hiện trôi rẻ chạy được trên ARM; (ii) cân bằng quên–học (stability–plasticity); (iii) open-world (lớp tấn công mới xuất hiện theo thời gian).

### Trục B — Học liên kết phân tán bảo mật (Federated Learning) cho IDS biên
- **Khoảng trống:** Nhiều tổ chức/thiết bị không chia sẻ được dữ liệu thô (riêng tư). FL cho phép huấn luyện cộng tác không lộ dữ liệu, nhưng còn vướng **non-IID**, **chi phí truyền thông**, và **đầu độc mô hình (poisoning)**.
- **Đóng góp NCS:** Khung FL cho IDS **trên cụm biên dị thể, có kiểm soát non-IID + chống đầu độc**, tích hợp tổng hợp an toàn (secure aggregation) — và quan trọng: *đánh giá loại-trừ-rò-rỉ trong bối cảnh phân tán* (mở rộng điểm mạnh của bạn sang FL, nơi rò rỉ còn ít được kiểm soát).
- **Kế thừa:** Bạn đã có cụm phân tán thật + Spark; nâng từ "phân tán huấn luyện một dataset" lên "phân tán nhiều bên giữ dữ liệu riêng".
- **Phương pháp:** FedAvg/FedProx + secure aggregation, robust aggregation chống Byzantine, mô phỏng non-IID theo lớp tấn công, kết hợp edge inference cho độ trễ thấp.
- **Bài toán con:** (i) FL + concept drift (kết hợp Trục A); (ii) đánh đổi riêng-tư/độ-chính-xác (differential privacy); (iii) phát hiện client độc.

### Trục C — Bền vững đối kháng (Adversarial Robustness & Evasion)
- **Khoảng trống:** ML-IDS dễ bị **evasion** (chỉnh lưu lượng để né) và **poisoning**. Đây chính là hạn chế mà luận văn của bạn *đã tự thừa nhận chưa làm*. Phòng thủ hiện tại thường giả định mô hình tấn công cố định; hướng mở là **đối thủ đa hướng, thích nghi**.
- **Đóng góp NCS:** Đánh giá + phòng thủ đối kháng **khả thi trên biên** (adversarial training, feature-squeezing, phát hiện perturbation) *dưới ràng buộc thời gian thực*, và xét **tính khả thi của evasion trong không gian lưu lượng mạng thật** (không phải chỉ trong không gian đặc trưng).
- **Kế thừa:** anomaly gate + ensemble của bạn là điểm khởi đầu để nghiên cứu "đối thủ né cả gate lẫn classifier".
- **Phương pháp:** GAN sinh traffic né tránh, adversarial training, certified/robustness bounds, ràng buộc khả thi của lưu lượng (giữ chức năng tấn công).
- **Bài toán con:** (i) ràng buộc khả thi miền mạng; (ii) robustness của pipeline 2 tầng (gate+classifier); (iii) kết hợp evasion + poisoning + backdoor.

### Trục D — Phát hiện zero-day: Tự giám sát, lưu lượng mã hoá, open-world
- **Khoảng trống:** Học có giám sát chỉ bắt tấn công **đã biết**; lưu lượng ngày càng **mã hoá** khiến đặc trưng payload vô dụng; cần phát hiện **zero-day**. Self-supervised/contrastive/transformer trên chuỗi gói thô đang nổi.
- **Đóng góp NCS:** Mô hình **tự giám sát/đối lập (contrastive)** học biểu diễn lưu lượng *không cần nhãn*, phát hiện bất thường **open-world** (lớp chưa từng thấy), chạy được trên biên — thay cho anomaly gate autoencoder đơn giản hiện tại.
- **Kế thừa:** autoencoder gate của bạn chính là phiên bản sơ khai của hướng này — nâng lên SSL/transformer/contrastive.
- **Phương pháp:** contrastive pretraining trên flow/packet, masked modeling, mô hình mở (open-set recognition), kết hợp diffusion/LLM cho lưu lượng mã hoá.
- **Bài toán con:** (i) biểu diễn không nhãn cho flow; (ii) ngưỡng open-set hiệu chỉnh theo drift; (iii) chi phí SSL trên ARM.

### Trục E — IDS thế hệ mới: LLM + Đồ thị (GNN) + XAI
- **Khoảng trống:** LLM đang được dùng cho phát hiện/giải thích/tự động hoá SOC; GNN khai thác cấu trúc đồ thị luồng; XAI cần thiết cho môi trường quan trọng. Thách thức: **ảo giác (hallucination)**, **chi phí trên biên**, bảo mật của chính LLM.
- **Đóng góp NCS:** Kiến trúc **lai LLM nhẹ + GNN + giải thích** cho NIDS: GNN nắm quan hệ host–flow, LLM diễn giải/ưu tiên cảnh báo cho phân tích viên, *đủ nhẹ để chạy biên* (lightweight LLM cho IoT là hướng mới 2026).
- **Kế thừa:** SHAP của bạn → nâng XAI thành phần cốt lõi; cảnh báo Email/Slack → tự động hoá có giải thích.
- **Phương pháp:** GNN trên đồ thị lưu lượng, LLM nhẹ/distilled, attribution-based explanation, chống hallucination bằng grounding.
- **Bài toán con:** (i) LLM nhẹ chạy Jetson; (ii) giải thích trung thực (faithful) chứ không hợp lý hoá; (iii) bảo mật bản thân LLM.

### Trục F — Khoa học đánh giá & dữ liệu đáng tin (đây là "ngách" mạnh nhất của bạn)
- **Khoảng trống:** Cộng đồng IDS vẫn vật lộn với **dữ liệu kém thực tế, rò rỉ, thiếu khả năng tái lập** (CICIDS2017 có nhiều lỗi nhãn/trùng lặp được ghi nhận liên tục 2018→2023). "Kỳ vọng vs thực tế" cho thấy độ chính xác cao trên dữ liệu tổng hợp *không* chuyển sang môi trường thật.
- **Đóng góp NCS:** Một **chuẩn đánh giá loại-trừ-rò-rỉ, có đối thủ thích nghi, tái lập được** cho IDS — biến điểm mạnh phương pháp luận của luận văn thành **đóng góp khoa học chính** của luận án: phát hiện rò rỉ tự động, sinh dữ liệu/đối thủ thực tế hơn, đo lường per-class công bằng, khung benchmark mở.
- **Kế thừa:** trực tiếp từ quy trình loại-trừ-rò-rỉ + kiểm định thống kê + cross-dataset của bạn.
- **Phương pháp:** kiểm toán rò rỉ tự động (temporal/near-duplicate/feature leakage), sinh lưu lượng tấn công thích nghi, đánh giá phân phối-thực tế, leaderboard tái lập.
- **Bài toán con:** (i) công cụ phát hiện rò rỉ tổng quát; (ii) bộ dữ liệu/CTF đánh giá tập trung evasion; (iii) thước đo "khoảng cách thực tế" (sim-to-real gap).

### Trục G — Hiệu quả phần cứng & đồng thiết kế (TinyML / quantization / năng lượng–chính xác–bền vững)
- **Khoảng trống:** Suy luận IDS trên thiết bị ràng buộc tài nguyên cần tối ưu mạnh (quantization, ONNX/TensorRT, pruning). Đánh đổi **năng lượng – độ trễ – độ chính xác – bền vững** chưa được tối ưu đồng thời.
- **Đóng góp NCS:** **Đồng thiết kế mô hình–phần cứng** cho IDS biên: lượng tử hoá/biên dịch TensorRT, khai thác GPU/Tensor core (mà luận văn chưa dùng), tối ưu đa mục tiêu năng lượng–chính xác–độ trễ–robustness.
- **Kế thừa:** baseline engine (Spark vs sklearn vs ONNX) + đo năng lượng tegrastats của bạn là *bước đầu hoàn hảo* cho hướng này.
- **Phương pháp:** post-training quantization, TensorRT, NAS nhận biết phần cứng, Pareto năng lượng–chính xác.
- **Bài toán con:** (i) lượng tử hoá không mất robustness; (ii) khai thác GPU Jetson; (iii) tối ưu Pareto đa mục tiêu trên biên.

---

## 3. Ba "câu chuyện luận án" gợi ý (cách gộp các trục thành một mạch nhất quán)

Một luận án mạnh cần **một câu hỏi xuyên suốt**, không phải 7 hướng rời rạc. Ba kịch bản:

**Kịch bản 1 (KHUYẾN NGHỊ) — "IDS biên đáng tin: thích ứng, bền vững, đánh giá trung thực".**
Gộp **A (học liên tục) + C (đối kháng) + F (đánh giá) + G (hiệu quả biên)**.
Mạch: *Một hệ IDS trên cụm biên vừa thích ứng trôi khái niệm, vừa bền trước đối kháng, được đánh giá dưới một chuẩn loại-trừ-rò-rỉ có đối thủ thích nghi, tối ưu cho phần cứng ràng buộc.*
→ **Tận dụng tối đa** hạ tầng + điểm mạnh phương pháp luận hiện có; rủi ro vừa; tính hệ thống cao; rất "khả thi tiến sĩ".

**Kịch bản 2 — "Học cộng tác bảo mật cho IDS phân tán".**
Gộp **B (federated) + A (drift) + C (poisoning/robustness)**.
Mạch: *FL cho IDS trên nhiều cụm biên giữ dữ liệu riêng, chịu non-IID + trôi + client độc, đánh giá loại-trừ-rò-rỉ trong môi trường phân tán.*
→ Rất hợp xu hướng riêng-tư + phân tán; cần thêm nhiều site/dữ liệu để thuyết phục.

**Kịch bản 3 (rủi ro/đột phá cao) — "Phát hiện zero-day không nhãn thế hệ mới".**
Gộp **D (self-supervised/open-world) + E (LLM/GNN/XAI) + F (đánh giá)**.
Mạch: *Biểu diễn tự giám sát + GNN + LLM nhẹ phát hiện tấn công chưa biết trên lưu lượng (kể cả mã hoá), giải thích được, đánh giá nghiêm ngặt.*
→ Trendy, tiềm năng công bố cao, nhưng rủi ro kỹ thuật và chi phí biên lớn hơn.

---

## 4. Khuyến nghị cụ thể

- **Chọn Kịch bản 1** nếu muốn tận dụng tối đa luận văn và có lộ trình chắc chắn: nó nối thẳng bốn tài sản hiện có, mỗi trục là một (vài) chương + một bài báo.
- **Giữ "đánh giá loại-trừ-rò-rỉ" (Trục F) làm sợi chỉ đỏ** xuyên suốt mọi trục — đây là chữ ký học thuật của bạn, ít người làm nghiêm túc, và là nơi bạn đã có lợi thế cạnh tranh.
- **Mở rộng dữ liệu/đối thủ:** bổ sung CSE-CIC-IDS2018 (đã có), CIC-IoT-2023, RoEduNet, và *lưu lượng thật/đối thủ thích nghi* để vượt giới hạn CICIDS2017.
- **Định vị trung thực về quy mô:** từ PoC 2 nút lên đánh giá khả mở thực sự (nhiều nút, lưu lượng trực tuyến) — chính là phần "future work" luận văn đã nêu.

---

## 5. Bước tiếp theo đề xuất

1. Chốt 1 kịch bản → viết **đề cương NCS** (research proposal) 8–12 trang: câu hỏi nghiên cứu, 3–4 mục tiêu, phương pháp, kế hoạch công bố.
2. Lập **bản đồ chương ↔ bài báo** (mỗi trục ≈ 1–2 bài hội nghị/tạp chí).
3. Khảo sát sâu (systematic review) cho trục chính → đây thường là **chương 2 + bài báo survey** đầu tiên của NCS.

*(Mình có thể giúp dựng đề cương NCS, bản đồ chương–bài báo, hoặc một systematic review cho trục bạn chọn.)*

---

## Nguồn tham khảo (đã kiểm chứng qua tìm kiếm 2024–2026)

- Deep Learning-based IDS: A Survey — arXiv:2504.07839 — https://arxiv.org/abs/2504.07839
- A comprehensive survey on DL-based IDS in IoT (Al-Haija, 2025, Expert Systems) — https://onlinelibrary.wiley.com/doi/abs/10.1111/exsy.13726
- Federated Learning-Based IDS in Industrial IoT (MDPI, 2026) — https://www.mdpi.com/1999-5903/18/1/2
- A survey of privacy-preserving federated learning for IDS (Springer, 2026) — https://link.springer.com/article/10.1007/s10462-026-11519-4
- Hybrid DL-FL IDS for IoT/5G edge (arXiv, 2025) — https://arxiv.org/html/2509.15555v1
- A Defensive Framework Against Adversarial Attacks on ML-NIDS — arXiv:2502.15561 — https://arxiv.org/abs/2502.15561
- Adversarial Defense in Cybersecurity: GANs review (arXiv, 2025) — https://arxiv.org/html/2509.20411v2
- A Review of the Duality of Adversarial Learning in Network Intrusion — arXiv:2412.13880 — https://arxiv.org/pdf/2412.13880
- CITADEL: Continual Anomaly Detection for IoT IDS — arXiv:2508.19450 — https://arxiv.org/pdf/2508.19450
- Continual Learning for IDS Under Evolving Network Threats (MDPI, 2025) — https://www.mdpi.com/1999-5903/17/10/456
- SOUL: Semi-supervised Open-world Continual Learning for NIDS — arXiv:2412.00911 — https://arxiv.org/pdf/2412.00911
- Evolving ML in Non-Stationary Environments: drift/forgetting survey — arXiv:2505.17902 — https://arxiv.org/pdf/2505.17902
- LLMs for NIDS: Foundations, Implementations, Future Directions — arXiv:2507.04752 — https://arxiv.org/html/2507.04752v1
- LLM + GNN + XAI for next-gen NIDS (Springer, 2025) — https://link.springer.com/article/10.1007/s10844-025-00964-2
- Lightweight LLMs for Network Attack Detection in IoT — arXiv:2601.15269 — https://arxiv.org/pdf/2601.15269
- Self-Supervised Transformer Contrastive Learning for IDS — arXiv:2505.08816 — https://arxiv.org/pdf/2505.08816
- Anomaly detection in encrypted traffic using SSL (Scientific Reports, 2025) — https://www.nature.com/articles/s41598-025-08568-0
- Troubleshooting an IDS Dataset: CICIDS2017 (Engelen et al.) — https://intrusion-detection.distrinet-research.be/WTMC2021/Resources/wtmc2021_Engelen_Troubleshooting.pdf
- Errors in the CICIDS2017 Dataset (Lanvin et al., Springer 2023) — https://link.springer.com/chapter/10.1007/978-3-031-31108-6_2
- Expectations Versus Reality: Evaluating IDS in Practice — arXiv:2403.17458 — https://arxiv.org/html/2403.17458v2
- StealthCup: Realistic, Evasion-Focused CTF for Benchmarking IDS — arXiv:2511.17761 — https://arxiv.org/pdf/2511.17761
