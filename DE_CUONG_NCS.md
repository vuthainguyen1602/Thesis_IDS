# ĐỀ CƯƠNG NGHIÊN CỨU SINH (PhD Research Proposal)

**Lĩnh vực:** An toàn thông tin / Học máy ứng dụng cho an ninh mạng / Tính toán biên

**Người soạn:** NCS (kế thừa luận văn thạc sĩ *"Phát hiện xâm nhập (IDS) dựa trên Apache Spark cho bảo mật mạng IoT"*)

**Ngày:** 25/06/2026

---

## 1. Tên đề tài

**Tên chính (đề xuất):**

> **Hệ thống phát hiện xâm nhập biên đáng tin cậy: thích ứng trôi khái niệm, bền vững đối kháng, và đánh giá loại-trừ-rò-rỉ.**
> *(Trustworthy Edge Intrusion Detection: Concept-Drift Adaptation, Adversarial Robustness, and Leakage-Free Evaluation.)*

**Phương án thay thế:**

- **PA1:** *"Học liên tục bền vững đối kháng cho phát hiện xâm nhập trên cụm biên ràng buộc tài nguyên, dưới khung đánh giá trung thực."*
- **PA2:** *"Từ độ chính xác đến độ tin cậy: một khung phương pháp luận cho IDS biên thích ứng, bền vững và tái lập được."*

Sợi chỉ đỏ xuyên suốt cả ba phương án là **khoa học đánh giá đáng tin (loại-trừ-rò-rỉ, có đối thủ thích nghi, tái lập được)** — chữ ký phương pháp luận kế thừa trực tiếp từ luận văn thạc sĩ — được áp dụng nhất quán cho cả ba năng lực kỹ thuật: thích ứng (A), bền vững (C) và hiệu quả biên (G).

---

## 2. Tóm tắt (Abstract)

Các hệ thống phát hiện xâm nhập (IDS) dựa trên học máy thường được công bố với độ chính xác gần tuyệt đối (F1 ≈ 1.0) trên các bộ dữ liệu chuẩn, nhưng độ chính xác đó phần lớn đến từ **rò rỉ dữ liệu** và đánh giá tĩnh, không phản ánh năng lực thực khi triển khai. Ba khoảng trống cốt lõi vẫn tồn tại: mô hình **suy giảm theo thời gian** trước trôi khái niệm, **dễ bị né tránh (evasion)** bởi kẻ tấn công chủ động, và khó **triển khai hiệu quả trên thiết bị biên** ràng buộc tài nguyên. Luận án này đề xuất một hệ IDS chạy trên **cụm biên dị thể (máy chủ điều phối + cụm Jetson Orin)**, kết hợp đồng thời ba năng lực: (i) **học liên tục** với phát hiện trôi rẻ và bộ đệm phát lại có chọn lọc nhằm chống quên thảm khốc dưới ràng buộc tài nguyên ARM; (ii) **huấn luyện và đánh giá đối kháng** xét tới ràng buộc khả thi trong không gian lưu lượng mạng thật, bao gồm cả độ bền của kiến trúc hai tầng cổng-bất-thường + bộ phân loại; (iii) **đồng thiết kế mô hình–phần cứng** (lượng tử hoá, biên dịch TensorRT, khai thác GPU/Tensor core) tối ưu Pareto năng lượng–độ trễ–chính xác–bền vững. Đóng góp phương pháp luận trung tâm là một **khung đánh giá loại-trừ-rò-rỉ, có đối thủ thích nghi và tái lập được**, dùng làm thước đo chung cho cả ba năng lực, mở rộng quy trình kiểm soát rò rỉ và kiểm định thống kê đã thiết lập ở luận văn thạc sĩ. Thực nghiệm dự kiến trên CICIDS2017, CSE-CIC-IDS2018 và CIC-IoT-2023 theo các kịch bản chia tách thời gian và liên-bộ-dữ-liệu. Mục tiêu là chuyển trọng tâm của IDS học máy *từ độ chính xác báo cáo sang độ tin cậy có thể kiểm chứng*.

*(~210 từ)*

---

## 3. Đặt vấn đề và động lực

### 3.1. Điểm tựa từ luận văn thạc sĩ

Luận văn thạc sĩ đã xây dựng và đánh giá một IDS cho mạng IoT trên nền **Apache Spark**, với bốn tài sản nền tảng có giá trị hiếm để mở rộng lên bậc tiến sĩ:

1. **Quy trình đánh giá loại-trừ-rò-rỉ + kiểm định thống kê:** loại đặc trưng rò rỉ nhãn (`destination_port`), khử trùng lặp ở mức đặc trưng, holdout tách rời, kiểm định hoán vị (permutation test), và đánh giá đa lớp. Đây là điểm khác biệt mạnh nhất, vì phần lớn công trình IDS vẫn báo F1 ≈ 1.0 do rò rỉ chưa kiểm soát.
2. **Hạ tầng huấn luyện phân tán thật:** cụm Spark gồm máy chủ Mac và **hai nút Jetson Orin Nano Super Developer Kit (8 GB)**, kèm cầu nối train→deploy (export `PipelineModel` ra biên).
3. **Pipeline biên thời gian thực:** Kafka + **anomaly gate (autoencoder)** + bộ phân loại, có đo năng lượng/độ trễ/throughput (tegrastats).
4. **Giải thích mô hình (SHAP)** và so sánh engine suy luận (Spark vs sklearn/ONNX).

### 3.2. Năm hạn chế cốt lõi cần giải quyết

Chính luận văn đã tự thừa nhận các hạn chế sau (Chương 5), tạo động lực trực tiếp cho luận án:

1. **Chỉ phát hiện tấn công đã biết, mô hình tĩnh:** Hệ thống giả định phân bố tấn công xấp xỉ CICIDS2017 và không cập nhật theo thời gian. Mối đe doạ thực tế tiến hoá liên tục, khiến mô hình tĩnh suy giảm — cần **thích ứng trôi khái niệm mà không quên thảm khốc** các tấn công cũ.
2. **Chưa đánh giá/bền vững trước lưu lượng đối kháng:** Luận văn ghi rõ "độ bền vững trước kẻ tấn công cố ý đánh lừa mô hình vẫn là câu hỏi mở". Kiến trúc hai tầng (gate + classifier) còn tạo thêm bề mặt tấn công: kẻ địch có thể né cả cổng lẫn bộ phân loại.
3. **Trần recall do anomaly gate:** Cổng bất thường đặt trước bộ phân loại tạo một trần cứng cho recall — các tấn công bị cổng bỏ sót sẽ không bao giờ tới bộ phân loại. Đây là **rủi ro an ninh**, không chỉ là đánh đổi tải.
4. **Quy mô và phần cứng biên mới ở mức minh chứng khái niệm:** Xử lý phân tán mới chứng minh trên 2 nút Jetson với vài triệu luồng; chưa khai thác GPU/Tensor core; chưa tối ưu lượng tử hoá/TensorRT; chưa tối ưu đa mục tiêu năng lượng–chính xác–bền vững.
5. **Phạm vi dữ liệu và giả định đánh giá còn hẹp:** Holdout đồng phân phối với tập kiểm tra (in-distribution); đánh giá chủ yếu trên CICIDS2017 (cùng với cặp liên-bộ CSE-CIC-IDS2018). Cần mở rộng sang dữ liệu mới hơn (CIC-IoT-2023) và **đối thủ thực tế thích nghi** để vượt giới hạn dữ liệu tổng hợp.

### 3.3. Động lực tổng quát

Cộng đồng IDS đang chuyển trọng tâm từ "độ chính xác cao trên benchmark" sang "độ tin cậy khi triển khai". Các khảo sát 2024–2026 chỉ ra rằng độ chính xác cao trên dữ liệu tổng hợp **không** chuyển sang môi trường thật, do rò rỉ, dữ liệu kém thực tế, thiếu khả năng tái lập, và thiếu đánh giá đối kháng/trôi (Apruzzese et al., 2024; Lanvin et al., 2023; Engelen et al., 2021). Luận án này nắm bắt đúng dịch chuyển đó: thay vì thêm một mô hình deep learning, nó **gộp bốn trục — học liên tục (A), đối kháng (C), khoa học đánh giá (F, làm sợi chỉ đỏ), và hiệu quả biên (G) — thành một câu hỏi xuyên suốt về độ tin cậy của IDS biên.**

---

## 4. Khoảng trống nghiên cứu (đối chiếu state-of-the-art 2024–2026)

**(a) Học liên tục/thích ứng trôi trên biên.** Học liên tục cho IDS là hướng nóng 2025–2026: CITADEL (arXiv:2508.19450) đề xuất phát hiện bất thường liên tục cho IDS IoT; SOUL (arXiv:2412.00911) xử lý học liên tục bán giám sát open-world; các khảo sát về trôi/quên (arXiv:2505.17902) và continual learning dưới mối đe doạ tiến hoá (MDPI Future Internet, 2025) cho thấy chủ đề trưởng thành. Các khung dựa trên ADWIN cho IDS streaming (Adaptive-Delta ADWIN, 2025; Trinity-Controller ADWIN, 2025) tinh chỉnh độ nhạy phát hiện trôi. **Khoảng trống:** hầu hết giả định tài nguyên server; rất ít công trình giải quyết đồng thời (i) phát hiện trôi **đủ rẻ để chạy trên ARM**, (ii) cân bằng ổn định–dẻo (stability–plasticity) dưới ngân sách bộ nhớ Jetson, và (iii) **open-world** khi lớp tấn công mới xuất hiện theo thời gian.

**(b) Bền vững đối kháng cho NIDS.** Các khung phòng thủ (arXiv:2502.15561) và khảo sát GAN-đối-kháng (arXiv:2509.20411) cùng tổng quan tính nhị nguyên của học đối kháng trong xâm nhập mạng (arXiv:2412.13880) đều phát triển mạnh. **Khoảng trống:** phần lớn tấn công/phòng thủ thực hiện trong **không gian đặc trưng**, bỏ qua **ràng buộc khả thi của lưu lượng mạng thật** (giữ chức năng tấn công, hợp lệ giao thức); rất ít công trình xét độ bền của **pipeline hai tầng (gate + classifier)** — đúng kiến trúc của luận văn; và đánh giá đối kháng hiếm khi đặt dưới ngân sách thời-gian-thực của biên.

**(c) Khoa học đánh giá và dữ liệu đáng tin (sợi chỉ đỏ).** CICIDS2017 có nhiều lỗi nhãn/trùng lặp được ghi nhận liên tục (Engelen et al., 2021; Lanvin et al., 2023); "Expectations vs Reality" (arXiv:2403.17458) định lượng khoảng cách sim-to-real; StealthCup (arXiv:2511.17761) đề xuất CTF tập trung né tránh để benchmark IDS thực tế hơn. **Khoảng trống:** thiếu **công cụ kiểm toán rò rỉ tự động, tổng quát** (temporal/near-duplicate/feature leakage), thiếu thước đo "khoảng cách thực tế", và thiếu một **leaderboard tái lập** kết hợp đồng thời rò rỉ + đối thủ thích nghi + đo per-class công bằng. Đây chính là nơi luận án có lợi thế cạnh tranh rõ rệt.

**(d) Hiệu quả phần cứng biên.** TensorRT/lượng tử hoá INT8–FP16 cho tốc độ gấp nhiều lần và năng lượng thấp hơn đáng kể trên Jetson Orin (TensorRT on edge SoC, NSF/Texas State, 2023; benchmark Jetson Orin NX, MDPI Computers 2026); chưng cất + lượng tử hoá cho phát hiện bất thường trên biên đã được khảo sát (arXiv:2407.02968). **Khoảng trống:** các nghiên cứu tối ưu biên chủ yếu cho thị giác máy tính; với IDS, **đánh đổi năng lượng–độ trễ–chính xác–bền vững chưa được tối ưu đồng thời (Pareto đa mục tiêu)**, và ảnh hưởng của lượng tử hoá lên **độ bền đối kháng** gần như chưa được đo.

**Tổng hợp khoảng trống:** chưa có công trình nào **gộp** thích ứng trôi + bền vững đối kháng + hiệu quả biên *dưới một khung đánh giá loại-trừ-rò-rỉ thống nhất*. Luận án định vị đúng vào giao điểm này.

---

## 5. Câu hỏi nghiên cứu

Mỗi câu hỏi ánh xạ một trục; **RQ3 (Trục F) là sợi chỉ đỏ** chi phối thước đo của cả ba RQ còn lại.

- **RQ1 (Trục A — Học liên tục/trôi):** Làm thế nào để IDS trên cụm biên ràng buộc tài nguyên **thích ứng trôi khái niệm** (kể cả lớp tấn công mới, open-world) mà **không quên thảm khốc** các tấn công đã biết, với chi phí phát hiện trôi và cập nhật mô hình đủ rẻ để chạy thời-gian-thực trên Jetson?

- **RQ2 (Trục C — Đối kháng):** IDS biên (đặc biệt kiến trúc hai tầng gate+classifier) **bền vững đến đâu** trước các tấn công né tránh **khả thi trong không gian lưu lượng mạng thật**, và phòng thủ nào (huấn luyện đối kháng, phát hiện nhiễu loạn) **khả thi dưới ngân sách thời-gian-thực của biên** mà không phá vỡ độ chính xác sạch và năng lực thích ứng (RQ1)?

- **RQ3 (Trục F — Đánh giá, sợi chỉ đỏ):** Một **khung đánh giá loại-trừ-rò-rỉ, có đối thủ thích nghi và tái lập được** cần những thành phần nào để đo *trung thực* năng lực IDS, và khung đó **thay đổi xếp hạng** mô hình/phòng thủ ra sao so với đánh giá tĩnh thông thường?

- **RQ4 (Trục G — Hiệu quả biên):** Có thể **đồng thiết kế mô hình–phần cứng** (lượng tử hoá, TensorRT, khai thác GPU/Tensor core) để đạt biên Pareto tốt giữa **năng lượng–độ trễ–chính xác–bền vững** trên Jetson, và lượng tử hoá ảnh hưởng thế nào tới độ bền đối kháng (RQ2) và năng lực thích ứng (RQ1)?

---

## 6. Mục tiêu và đóng góp dự kiến

### MT1 — Khung học liên tục thích ứng trôi trên biên (đáp RQ1, Trục A)

Xây dựng cơ chế học tăng tiến với phát hiện trôi rẻ + bộ đệm phát lại có chọn lọc + chưng cất tri thức, vận hành dưới ngân sách bộ nhớ/tính toán của Jetson, hỗ trợ open-world.

- **Đóng góp khoa học mới:** Khung học liên tục **đồng-thiết-kế-với-ràng-buộc-biên** với chiến lược chọn mẫu phát lại tối ưu cho ngân sách ARM và cơ chế nhận diện lớp mới có hiệu chỉnh ngưỡng theo trôi — điều các công trình trên server bỏ qua. Pipeline streaming Kafka + cụm Jetson hiện có là *testbed tự nhiên*.

### MT2 — Đánh giá và phòng thủ đối kháng khả thi miền-mạng cho pipeline hai tầng (đáp RQ2, Trục C)

Thiết lập mô hình kẻ địch có **ràng buộc khả thi lưu lượng**, đánh giá độ bền của kiến trúc gate+classifier, và phát triển phòng thủ chạy được trên biên.

- **Đóng góp khoa học mới:** Phân tích đầu tiên (theo hiểu biết hiện tại) về **độ bền đối kháng của pipeline hai tầng anomaly-gate + classifier** dưới ràng buộc khả thi miền-mạng, kèm phòng thủ tôn trọng ngân sách thời-gian-thực — biến chính điểm yếu "trần recall do gate" của luận văn thành đối tượng nghiên cứu định lượng.

### MT3 — Khung đánh giá loại-trừ-rò-rỉ, đối thủ thích nghi, tái lập (đáp RQ3, Trục F — trung tâm)

Phát triển bộ công cụ kiểm toán rò rỉ tự động (temporal/near-duplicate/feature leakage), giao thức đánh giá có đối thủ thích nghi, đo per-class công bằng, và một **leaderboard tái lập** (mã/đường-dẫn-dữ-liệu/seed cố định).

- **Đóng góp khoa học mới:** Biến quy trình loại-trừ-rò-rỉ thủ công của luận văn thành **công cụ tự động, tổng quát hoá nhiều bộ dữ liệu**, kèm **thước đo khoảng cách thực tế (sim-to-real gap)** và bằng chứng định lượng cho việc **rò rỉ đảo xếp hạng mô hình**. Đây là *đóng góp khoa học chính* và là thước đo chung của toàn luận án.

### MT4 — Đồng thiết kế mô hình–phần cứng Pareto đa mục tiêu trên Jetson (đáp RQ4, Trục G)

Lượng tử hoá (PTQ/QAT), biên dịch TensorRT, khai thác GPU/Tensor core, và tối ưu Pareto năng lượng–độ trễ–chính xác–bền vững.

- **Đóng góp khoa học mới:** Bản đồ Pareto **bốn mục tiêu** cho IDS biên (lần đầu gắn **độ bền đối kháng** vào đánh đổi lượng-tử-hoá), và đo định lượng tác động của lượng tử hoá lên cả robustness (RQ2) lẫn khả năng học liên tục (RQ1). Mở rộng baseline engine (Spark vs sklearn vs ONNX) của luận văn sang TensorRT/INT8–FP16.

---

## 7. Phương pháp nghiên cứu theo từng mục tiêu

### 7.1. Dữ liệu và hạ tầng chung

- **Dữ liệu:** CICIDS2017 (nền), **CSE-CIC-IDS2018** (đã có cặp liên-bộ), **CIC-IoT-2023** (mở rộng độ thực tế IoT). Mọi bộ chạy qua module kiểm toán rò rỉ của MT3 trước khi dùng. Các kịch bản chia tách: ngẫu nhiên (baseline), **theo thời gian** (`temporal`, cho trôi), và **liên-bộ-dữ-liệu** (cross-dataset, cho tổng quát hoá).
- **Hạ tầng:** cụm biên dị thể = máy chủ điều phối (Mac/x86) + **cụm Jetson Orin**; điều phối/streaming qua Kafka; đo năng lượng/độ trễ/throughput bằng tegrastats; nhật ký vào hạ tầng giám sát (PostgreSQL/InfluxDB/Grafana) đã dựng ở luận văn, có bật xác thực/mã hoá.

### 7.2. Phương pháp cho MT1 (Học liên tục — Trục A)

- **Phát hiện trôi:** so sánh các bộ phát hiện nhẹ (ADWIN/DDM/biến thể dựa-lỗi) theo trục chi-phí-trên-ARM; chọn cấu hình Pareto độ-nhạy/chi-phí.
- **Học tăng tiến chống quên:** bộ đệm phát lại (replay) có chọn lọc dưới ngân sách bộ nhớ cố định; chưng cất tri thức (knowledge distillation) giữa phiên; chuẩn hoá thích ứng (adaptive normalization) cho dịch chuyển phân phối.
- **Open-world:** nhận diện lớp mới (open-set recognition) với ngưỡng hiệu chỉnh theo tín hiệu trôi.
- **Đánh giá:** độ chính xác theo thời gian, **độ quên ngược (backward transfer)**, độ dẻo (forward transfer), thời gian/năng lượng cập nhật trên Jetson — tất cả dưới khung MT3.

### 7.3. Phương pháp cho MT2 (Đối kháng — Trục C)

- **Mô hình kẻ địch:** tấn công né tránh trong không gian đặc trưng **có chiếu về ràng buộc khả thi lưu lượng** (giữ hợp lệ giao thức và chức năng tấn công); có thể dùng sinh lưu lượng đối kháng (GAN/ràng buộc) để kiểm chứng tính khả thi.
- **Mục tiêu kép:** đánh giá riêng độ bền của **cổng bất thường** và của **bộ phân loại**, và độ bền của **toàn pipeline hai tầng** (kẻ địch né cả hai).
- **Phòng thủ:** huấn luyện đối kháng, phát hiện nhiễu loạn/feature-squeezing, lựa chọn phòng thủ **dưới ngân sách độ trễ biên**; phân tích đánh đổi với độ chính xác sạch và với MT1.
- **Đánh giá:** robust accuracy/recall per-class dưới nhiều cường độ tấn công, chi phí thời-gian-thực — dưới khung MT3.

### 7.4. Phương pháp cho MT3 (Đánh giá — Trục F, sợi chỉ đỏ)

- **Kiểm toán rò rỉ tự động:** module phát hiện (i) rò rỉ nhãn ở mức đặc trưng (vd `destination_port`), (ii) trùng lặp/gần-trùng giữa train–test (near-duplicate), (iii) rò rỉ thời gian (temporal); báo cáo mức **thổi phồng hiệu năng** do rò rỉ.
- **Giao thức đối thủ thích nghi:** tích hợp tấn công né tránh (từ MT2) như một phần bắt buộc của đánh giá; đề xuất kịch bản kiểu **StealthCup** (CTF tập trung né tránh).
- **Đo per-class công bằng + kiểm định thống kê:** mở rộng permutation test/đa lớp của luận văn; thước đo **sim-to-real gap** qua đánh giá liên-bộ.
- **Tái lập:** đóng gói mã + cấu hình + seed; xuất **leaderboard tái lập**; mục tiêu phụ là phát hành công cụ kiểm toán dưới dạng mã nguồn mở.
- **Bằng chứng cốt lõi:** định lượng **rò rỉ làm đảo xếp hạng** mô hình/phòng thủ ra sao.

### 7.5. Phương pháp cho MT4 (Hiệu quả biên — Trục G)

- **Lượng tử hoá:** post-training quantization (INT8) và quantization-aware training; **biên dịch TensorRT**, khai thác **Tensor core** (FP16) trên Jetson Orin.
- **Tối ưu đa mục tiêu:** dựng biên **Pareto năng lượng–độ trễ–chính xác–bền vững**; (tuỳ chọn) tìm kiếm kiến trúc nhận-biết-phần-cứng.
- **Đo chéo:** tác động của lượng tử hoá lên **độ bền đối kháng** (RQ2) và lên **khả năng học liên tục** (RQ1) — đây là phần đo chéo ít công trình làm.
- **So sánh engine:** mở rộng Spark vs sklearn vs ONNX (luận văn) sang **TensorRT/INT8–FP16**.

---

## 8. Kế hoạch đánh giá và tiêu chí thành công

**Trục đo lường:**

- **Hiệu năng phân loại (per-class):** Precision/Recall/F1 theo từng lớp tấn công (không chỉ nhị phân), nhấn mạnh lớp hiếm; mọi con số đo **dưới cấu hình loại-trừ-rò-rỉ**.
- **Thích ứng (A):** độ chính xác-theo-thời-gian; backward/forward transfer; độ trễ phát hiện trôi; chi phí cập nhật (thời gian + năng lượng) trên Jetson.
- **Bền vững (C):** robust F1/recall dưới nhiều loại/cường độ tấn công né tránh khả thi; độ bền riêng của gate, classifier, và pipeline.
- **Hiệu quả biên (G):** độ trễ (ms/luồng), throughput, **năng lượng/suy-luận (J)** đo bằng tegrastats; điểm trên biên Pareto.
- **Tái lập (F):** tỷ lệ kết quả tái lập từ artifact công bố; mức thổi phồng do rò rỉ; sim-to-real gap qua cross-dataset.

**Tiêu chí thành công (định lượng hoá khi có baseline, tránh thổi phồng):**

1. Khung đánh giá MT3 **phát hiện được** rò rỉ trên ≥3 bộ dữ liệu và **chứng minh đảo xếp hạng** trên ít nhất một cặp mô hình.
2. Học liên tục (MT1) giảm **độ quên** so với fine-tuning ngây thơ ở mức có ý nghĩa thống kê, với chi phí cập nhật nằm trong ngân sách thời-gian-thực Jetson.
3. Phòng thủ đối kháng (MT2) cải thiện robust F1 so với mô hình không phòng thủ mà **không** làm sụt độ chính xác sạch quá ngưỡng thoả thuận, dưới ngân sách độ trễ biên.
4. MT4 đạt **giảm năng lượng/độ trễ đáng kể** (vd nhờ INT8/TensorRT) với mức mất chính xác/bền-vững được báo cáo minh bạch trên biên Pareto.
5. Tất cả thí nghiệm **tái lập được** từ artifact công bố.

*Lưu ý trung thực:* các ngưỡng cụ thể (vd "X% năng lượng", "ΔF1 ≤ Y") sẽ được chốt sau khi đo baseline ở Năm 1; đề cương cố ý không cam kết con số chưa đo được.

---

## 9. Kế hoạch công bố

Mỗi trục ánh xạ 1–2 bài; sợi chỉ đỏ F xuất hiện ở mọi bài và có một bài survey/benchmark riêng.

| # | Nội dung | Trục | Loại venue gợi ý (thực tế) |
|---|----------|------|-----------------------------|
| P1 | Survey/SLR: đánh giá đáng tin cho IDS học máy (rò rỉ, trôi, đối kháng, biên) | F (khung) | Tạp chí khảo sát: *ACM Computing Surveys*, *IEEE Communications Surveys & Tutorials* |
| P2 | Khung + công cụ đánh giá loại-trừ-rò-rỉ, đối thủ thích nghi, tái lập | F | *USENIX Security*, *ACM CCS*, *NDSS*; hoặc *RAID*, *DIMVA* |
| P3 | Học liên tục thích ứng trôi trên cụm biên Jetson | A | *ESORICS*, *ACM AsiaCCS*; hoặc *IEEE TIFS*, *Future Internet* (MDPI) |
| P4 | Bền vững đối kháng cho pipeline hai tầng, ràng buộc khả thi miền-mạng | C | *ACM AISec* (workshop CCS), *IEEE S&P* workshops; tạp chí: *Computers & Security* |
| P5 | Đồng thiết kế mô hình–phần cứng, Pareto năng lượng–chính xác–bền vững | G | *ACM/IEEE IoTDI*, *EuroSys*/edge workshops; tạp chí: *IEEE IoT Journal*, *IEEE TC* |
| P6 (tuỳ chọn) | Bài tổng hợp hệ thống: IDS biên đáng tin (gộp A+C+F+G) | A+C+F+G | Tạp chí Q1: *IEEE TDSC*, *IEEE TIFS* |

Mục tiêu thực tế: **4–6 bài** (≥2 hội nghị hạng A về an ninh và ≥1 tạp chí Q1), trong đó P1 đóng vai chương tổng quan của luận án.

---

## 10. Lộ trình 3–3.5 năm

**Năm 1 — Nền tảng + Trục F (sợi chỉ đỏ).**
- Q1: Khảo sát hệ thống (SLR) → bản thảo **P1**; chốt baseline trên CICIDS2017/2018/CIC-IoT-2023.
- Q2: Dựng module kiểm toán rò rỉ tự động (MT3 lõi); đo mức thổi phồng do rò rỉ.
- Q3: Giao thức đối thủ thích nghi + per-class + tái lập; bản thảo **P2**.
- Q4: Hoàn thiện leaderboard tái lập; nộp **P2**; nâng cấp hạ tầng biên (xác thực/mã hoá).

**Năm 2 — Trục A + Trục C.**
- Q1–Q2: Học liên tục + phát hiện trôi trên Jetson (MT1); bản thảo **P3**.
- Q3–Q4: Đánh giá/phòng thủ đối kháng pipeline hai tầng (MT2); bản thảo **P4**; nộp **P3**.

**Năm 3 — Trục G + tích hợp.**
- Q1: Lượng tử hoá/TensorRT + Pareto đa mục tiêu (MT4); đo chéo robustness↔lượng-tử-hoá.
- Q2: Tích hợp toàn hệ (A+C+F+G) chạy end-to-end trên cụm biên; nộp **P4**, **P5**.
- Q3: Đánh giá tổng hợp; bản thảo **P6**.

**Năm 3.5 (nửa năm đệm) — Hoàn thiện.**
- Q1: Nộp **P6**; viết luận án.
- Q2: Bảo vệ; phát hành artifact mã nguồn mở.

**Mốc chính:** M1 (cuối Năm 1) — công cụ đánh giá + P1/P2; M2 (cuối Năm 2) — A & C hoạt động trên biên + P3/P4; M3 (giữa Năm 3) — G + hệ tích hợp; M4 (Năm 3.5) — bảo vệ.

---

## 11. Rủi ro và giảm thiểu

| Rủi ro | Mức | Giảm thiểu |
|--------|-----|-----------|
| Ràng buộc tài nguyên Jetson khiến học liên tục/đối kháng quá chậm | Cao | Đồng-thiết-kế-với-biên ngay từ đầu (ngân sách bộ nhớ/độ trễ là ràng buộc thiết kế, không phải hậu kiểm); chưng cất + lượng tử hoá sớm (MT4 hỗ trợ MT1/MT2). |
| Tấn công đối kháng "khả thi miền-mạng" khó hiện thực hoá | Cao | Khởi đầu bằng ràng buộc đặc trưng đơn giản, tăng dần độ chân thực; hợp tác/đối chiếu kịch bản kiểu StealthCup; trung thực về phạm vi khả thi đã kiểm chứng. |
| Dữ liệu công khai vẫn kém thực tế dù đã khử rò rỉ | Trung bình | Đánh giá liên-bộ + thước đo sim-to-real; trình bày kết quả như cận-trên/cận-dưới, không tổng quát hoá quá mức. |
| Phạm vi luận án quá rộng (4 trục) | Trung bình | Trục F là sợi chỉ đỏ ràng buộc; mỗi trục là một bài độc lập; ưu tiên cắt phạm vi G/A trước khi cắt F. |
| Khó tái lập/phụ thuộc phiên bản TensorRT/JetPack | Trung bình | Cố định container/seed/artifact; báo cáo cấu hình phần cứng-phần mềm đầy đủ. |
| Không đạt SOTA về độ chính xác thô | Thấp | Đóng góp chính là **độ tin cậy/phương pháp luận**, không phải bảng xếp hạng F1; định vị trung thực. |

---

## 12. Tài liệu tham khảo

*(Ưu tiên 2023–2026; kế thừa nguồn đã kiểm chứng trong `HUONG_PHAT_TRIEN_NCS.md`, bổ sung qua tìm kiếm 2025–2026.)*

**Khảo sát nền & IDS học sâu**
1. Deep Learning-based IDS: A Survey. arXiv:2504.07839. https://arxiv.org/abs/2504.07839
2. Al-Haija et al. A comprehensive survey on DL-based IDS in IoT (2025), *Expert Systems*. https://onlinelibrary.wiley.com/doi/abs/10.1111/exsy.13726

**Học liên tục / thích ứng trôi (Trục A)**
3. CITADEL: Continual Anomaly Detection for IoT IDS. arXiv:2508.19450. https://arxiv.org/pdf/2508.19450
4. Continual Learning for IDS Under Evolving Network Threats (2025), *Future Internet* (MDPI). https://www.mdpi.com/1999-5903/17/10/456
5. SOUL: Semi-supervised Open-world Continual Learning for NIDS. arXiv:2412.00911. https://arxiv.org/pdf/2412.00911
6. Evolving ML in Non-Stationary Environments: drift/forgetting survey. arXiv:2505.17902. https://arxiv.org/pdf/2505.17902
7. Adaptive-Delta ADWIN: Stable and Sensitive Intrusion Detection in Streaming Networks (2025), *J. of Information Systems and Informatics*. https://journal-isi.org/index.php/isi/article/view/1336
8. Trinity-Controller ADWIN: Accuracy-Guided Sensitivity Control for Streaming Intrusion Detection (2025), *J. of Information Systems and Informatics*. https://journal-isi.org/index.php/isi/article/view/1421

**Bền vững đối kháng (Trục C)**
9. A Defensive Framework Against Adversarial Attacks on ML-NIDS. arXiv:2502.15561. https://arxiv.org/abs/2502.15561
10. Adversarial Defense in Cybersecurity: GANs review (2025). arXiv:2509.20411. https://arxiv.org/html/2509.20411v2
11. A Review of the Duality of Adversarial Learning in Network Intrusion. arXiv:2412.13880. https://arxiv.org/pdf/2412.13880

**Khoa học đánh giá & dữ liệu đáng tin (Trục F — sợi chỉ đỏ)**
12. Engelen et al. Troubleshooting an IDS Dataset: CICIDS2017 (WTMC 2021). https://intrusion-detection.distrinet-research.be/WTMC2021/Resources/wtmc2021_Engelen_Troubleshooting.pdf
13. Lanvin et al. Errors in the CICIDS2017 Dataset (Springer, 2023). https://link.springer.com/chapter/10.1007/978-3-031-31108-6_2
14. Apruzzese et al. Expectations Versus Reality: Evaluating IDS in Practice. arXiv:2403.17458. https://arxiv.org/html/2403.17458v2
15. StealthCup: Realistic, Evasion-Focused CTF for Benchmarking IDS. arXiv:2511.17761. https://arxiv.org/pdf/2511.17761

**Hiệu quả phần cứng biên (Trục G)**
16. Zhou et al. TensorRT Implementations of Model Quantization on Edge SoC (MCSoC 2023). https://par.nsf.gov/servlets/purl/10488646
17. Benchmarking YOLOv8 Variants on Jetson Orin NX for Edge Computing (2026), *Computers* (MDPI). https://www.mdpi.com/2073-431X/15/2/74
18. Unified Anomaly Detection on Edge Device using Knowledge Distillation and Quantization. arXiv:2407.02968. https://arxiv.org/pdf/2407.02968
19. Quantized Object Detection for Real-Time Inference on Edge (2025), *IJACSA* 16(5). https://thesai.org/Downloads/Volume16No5/Paper_3-Quantized_Object_Detection_for_Real_Time_Inference.pdf

**Liên kết/phân tán & hướng mới (bối cảnh, không trọng tâm)**
20. Federated Learning-Based IDS in Industrial IoT (2026), *Future Internet* (MDPI). https://www.mdpi.com/1999-5903/18/1/2
21. A survey of privacy-preserving federated learning for IDS (Springer, 2026). https://link.springer.com/article/10.1007/s10462-026-11519-4
22. LLMs for NIDS: Foundations, Implementations, Future Directions. arXiv:2507.04752. https://arxiv.org/html/2507.04752v1
23. Self-Supervised Transformer Contrastive Learning for IDS. arXiv:2505.08816. https://arxiv.org/pdf/2505.08816
24. Anomaly detection in encrypted traffic using SSL (2025), *Scientific Reports*. https://www.nature.com/articles/s41598-025-08568-0
25. Lightweight LLMs for Network Attack Detection in IoT. arXiv:2601.15269. https://arxiv.org/pdf/2601.15269

---

*Ghi chú phương pháp luận: Đề cương cố ý không cam kết các con số hiệu năng chưa đo. Đóng góp chính là **độ tin cậy có thể kiểm chứng** của IDS biên (thích ứng + bền vững + đánh giá trung thực + hiệu quả phần cứng), kế thừa trực tiếp bốn tài sản của luận văn thạc sĩ và các hạn chế mà luận văn đã tự nhận diện.*
