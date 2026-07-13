# Tham khảo luận án tiến sĩ gần đây (mảng IDS / học máy / biên) — định hướng cho NCS

*Tài liệu tổng hợp để đối chiếu phạm vi, cấu trúc và đầu ra của các luận án/công trình tiến sĩ gần đây cùng mảng, làm cơ sở cho kế hoạch NCS (xem kèm `HUONG_PHAT_TRIEN_NCS.md`, `DE_CUONG_NCS.md`, `BAN_DO_CHUONG_BAI_BAO.md`). Nguồn đã tra/xác minh qua web (2021–2026).*

---

## 1. Vì sao xem các luận án này

Để (a) biết một luận án TS mảng IDS/ML "trông như thế nào" (phạm vi, số chương đóng góp, số bài báo), và (b) định vị ngách của bạn so với *frontier* hiện tại. Kết luận ngắn: **ngách "đánh giá loại-trừ-rò-rỉ + triển khai biên thật" của bạn đang là chủ đề nóng 2025–2026, không lỗi thời** — và hướng nâng tầm rõ nhất là chuyển từ *phân loại từng-luồng* sang *mô hình ngữ cảnh (context-aware: chuỗi thời gian + đồ thị)*.

---

## 2. Các luận án / công trình tiêu biểu gần đây

| Công trình | Tác giả / Đơn vị / Năm | Trọng tâm | Điểm rút ra cho bạn |
|---|---|---|---|
| **Deep Learning for Contextualized NetFlow-Based NIDS: Methods, Data, Evaluation and Deployment** (arXiv 2602.05594) | El Mahdaouy và cộng sự — Mohammed VI Polytechnic / Hassan II Univ. (Morocco), 2026 | Tổng hợp **context-aware** NIDS theo 4 chiều (thời gian, đồ thị/quan hệ, đa phương thức, đa độ phân giải); nhấn mạnh **đánh giá nghiêm ngặt** + **triển khai** | **Xác nhận trực tiếp ngách của bạn**: nêu thẳng rò rỉ thời gian, chia dữ liệu sai, lỗi nhãn CIC-IDS-2017/2018 (Engelen, Lanvin), kém khái quát hoá liên-bộ; và ràng buộc triển khai (trạng thái luồng, bộ nhớ, độ trễ, nén mô hình). Đây là "bản đồ" frontier để bạn cắm cờ |
| **Intrusion Detection Systems using Machine Learning** | Hanan Hindy — Abertay University (UK), PhD 2021 | IDS huấn luyện với **dữ liệu hạn chế**, phát hiện zero-day, taxonomy mối đe doạ | Mẫu một luận án TS hoàn chỉnh (toàn văn công khai): cách dựng taxonomy, chương đóng góp, đánh giá |
| **Novel applications of ML to Network Traffic Analysis and Prediction** | PhD thesis (công khai trên ResearchGate) | Phân loại lưu lượng + **sinh lưu lượng tấn công tổng hợp** + IDS | Gợi ý hướng dữ liệu tổng hợp/đối kháng (liên quan Trục C, F) |
| **Kho luận án ML — CMU** | Carnegie Mellon, Machine Learning Dept. | Tập hợp luận án TS ngành ML | Tham chiếu *chuẩn mực* cấu trúc/độ sâu/đầu ra của luận án TS hàng đầu |
| **Đề tài IDS tại UIT.NC (VN)** | UIT — ĐHQG TP.HCM (đồ án/luận văn) | IDS bằng **federated learning**, **homomorphic encryption**, IDS nhúng IoT | Cho thấy mảng trong nước đang đi vào FL/riêng-tư/biên — *trùng hướng Trục B*; phần lớn là thạc sĩ/đồ án, **luận án TS công khai còn ít** |

> Ghi chú liêm chính: bản El Mahdaouy 2026 là *synthesis/survey* nhiều tác giả (định dạng tiêu đề kiểu luận án), không phải luận án đơn-tác-giả — nhưng nội dung phản ánh đúng frontier. Các mục còn lại là luận án/đồ án thực, đã kiểm tra link.

---

## 3. Mẫu cấu trúc luận án TS mảng này ("thesis by publication")

Phổ biến nhất:

1. **Chương 1 — Mở đầu**: vấn đề, câu hỏi nghiên cứu, đóng góp, cấu trúc.
2. **Chương 2 — Tổng quan / Systematic review** *(≈ 1 bài survey)*.
3. **Chương 3–6 — Các chương đóng góp**, mỗi chương ≈ **1 bài báo** (một đóng góp khoa học độc lập).
4. **Chương 7 — Tích hợp / triển khai hệ thống** *(≈ 1 bài tạp chí tổng kết)*.
5. **Chương 8 — Kết luận & hướng phát triển**.

- **Đầu ra điển hình:** 3–5 bài, trong đó ≥1–2 tạp chí mạnh (WoS/Scopus) là tác giả chính.
- **Khớp kế hoạch của bạn:** 7 bài trong `BAN_DO_CHUONG_BAI_BAO.md` và mix **1 Q1 + 2 Q3 + 2 trong nước** → đúng hoặc trên mức mẫu này.

---

## 4. Bài học cho luận án của BẠN

1. **Ngách được xác nhận là frontier.** Bản tổng hợp mới nhất (2026) đặt *đánh giá loại-trừ-rò-rỉ + lỗi CICIDS2017/2018 + cross-dataset + deployment* làm trục chính → **Trục F của bạn đang nóng**, và bạn có lợi thế *đã làm thật* (code, ablation cổng rò rỉ, cross-dataset, đo năng lượng biên).
2. **Hướng nâng tầm rõ ràng:** chuyển từ **phân loại từng-luồng** (đúng cái luận văn thạc sĩ làm) sang **mô hình ngữ cảnh**:
   - *Thời gian*: RNN/Transformer trên chuỗi luồng/phiên (bắt beaconing, exfiltration nhiều giai đoạn).
   - *Đồ thị*: GNN trên đồ thị host–flow (bắt quét phối hợp, lateral movement).
   - Gắn tự nhiên với **Trục A (trôi khái niệm)** và **Trục C (đối kháng)**.
3. **Khoảng trống để chiếm:** gần như chưa công trình nào **gộp đồng thời** *context-aware + đánh giá loại-trừ-rò-rỉ nghiêm ngặt + triển khai biên thật (Jetson) + năng lượng + (federated/đối kháng)*. Đây chính là **Kịch bản 1** của bạn — vá đúng các "failure mode" mà frontier đang than phiền.
4. **Thông điệp bán hàng của luận án:** *"Không chỉ đề xuất mô hình tốt hơn, mà chứng minh nó tốt hơn dưới một quy trình đánh giá đáng tin (loại-trừ-rò-rỉ, cross-dataset, đối kháng) và triển khai được trên phần cứng biên thật."*

---

## 5. Cách tìm thêm luận án TS (đặc biệt Việt Nam)

- **Quốc tế:** Google Scholar + lọc "PhD thesis"/"dissertation"; kho trường (Abertay RKE, MIT DSpace, TU Delft Repository); ProQuest Dissertations; arXiv (cs.CR).
- **Việt Nam:** thư viện số trường (IUH, ĐHQG-HCM/HN), **Cơ sở dữ liệu luận án/luận văn của Bộ GD&ĐT**, trang Sau đại học của khoa CNTT/ATTT. Tra theo từ khoá "luận án tiến sĩ phát hiện xâm nhập", "học sâu an ninh mạng", "học liên kết IDS".
- **Mẹo:** xem **mục lục + phần Kết luận/Đóng góp + danh mục công trình đã công bố** của 2–3 luận án để chuẩn hoá kỳ vọng đầu ra và cách đóng gói đóng góp.

---

## Nguồn (đã tra/xác minh)

- El Mahdaouy et al. (2026) — Contextualized NetFlow-Based NIDS — https://arxiv.org/pdf/2602.05594
- Hanan Hindy — PhD Thesis 2021, Abertay University — https://rke.abertay.ac.uk/files/33845351/Hindy_IntrusionDetectionSystemsUsingMachineLearning_PhD_2021.pdf
- PhD Thesis — Novel applications of ML to Network Traffic Analysis and Prediction — https://www.researchgate.net/publication/336197910_PhD_Thesis_Novel_applications_of_Machine_Learning_to_Network_Traffic_Analysis_and_Prediction
- CMU — Machine Learning PhD Dissertations — https://ml.cmu.edu/research/phd-dissertations
- UIT.NC — Phát hiện tấn công cho hệ thống nhúng IoT bằng học liên kết — https://nc.uit.edu.vn/do-an/phat-hien-tan-cong-cho-he-thong-nhung-iot-bang-hoc-lien-ket
- (Nền dữ liệu/rò rỉ) Engelen et al. 2021; Lanvin et al. 2023; Arp et al. "Dos and Don'ts of ML in Security" — xem `SYSTEMATIC_REVIEW_DanhGia_IDS.md`
