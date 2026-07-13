# Tổng quan hệ thống: Đánh giá đáng tin và loại-trừ-rò-rỉ cho hệ thống phát hiện xâm nhập học máy trên thiết bị biên

*Systematic review làm nền cho Chương 2 và bài báo survey đầu tiên của NCS. Trục chính (Kịch bản 1): "Đánh giá đáng tin và loại-trừ-rò-rỉ cho hệ thống phát hiện xâm nhập học máy trên thiết bị biên", phủ các nhánh liền kề: rò rỉ dữ liệu và chất lượng bộ dữ liệu (CICIDS2017 và hậu duệ), đánh giá per-class/thống kê, thích ứng trôi khái niệm, bền vững đối kháng, và ràng buộc triển khai biên.*

*Phiên bản: 2026-06-25. Mọi nguồn trong bài đều được tra cứu và xác minh qua tìm kiếm web; DOI/URL ghi rõ ở mục 7. Không có công trình nào được bịa.*

---

## 1. Tóm tắt và động lực

### 1.1. Tóm tắt

Học máy (ML) và học sâu (DL) đã trở thành phương pháp chủ đạo cho hệ thống phát hiện xâm nhập mạng (NIDS). Tuy nhiên, một nghịch lý dai dẳng tồn tại: rất nhiều công trình báo cáo độ chính xác và F1-score gần như tuyệt đối (≈ 0.99–1.00) trên các bộ dữ liệu chuẩn như CICIDS2017, nhưng kết quả đó hầu như **không chuyển sang** môi trường vận hành thật. Nguyên nhân cốt lõi không nằm ở mô hình, mà ở **quy trình đánh giá**: rò rỉ dữ liệu (data leakage), lỗi nhãn và lỗi trích đặc trưng trong bộ dữ liệu, thiên lệch không-gian/thời-gian (spatial/temporal bias), và giả định phân phối tĩnh.

Tổng quan này hệ thống hoá tài liệu 2018–2026 về **đánh giá đáng tin (trustworthy evaluation)** cho NIDS học máy, với trọng tâm là **loại-trừ-rò-rỉ**, và mở rộng sang các điều kiện làm cho một đánh giá là "thật": tính đa lớp/thống kê, khái quát hoá liên bộ dữ liệu, thích ứng trôi khái niệm, bền vững đối kháng, và ràng buộc triển khai biên. Mục tiêu là định vị một ngách nghiên cứu cho NCS: biến quy trình đánh giá loại-trừ-rò-rỉ từ một "thực hành tốt" thành một **đóng góp phương pháp luận chính** — một chuẩn đánh giá đáng tin, có đối thủ thích nghi, tái lập được, và khả thi trên phần cứng biên ràng buộc.

### 1.2. Vì sao "đánh giá đáng tin" là vấn đề cốt lõi

Một hệ IDS chỉ có giá trị nếu **con số báo cáo phản ánh đúng năng lực thực**. Có ba lý do khiến đánh giá trở thành vấn đề cốt lõi chứ không phải vấn đề phụ:

1. **Rò rỉ làm hỏng kết luận một cách hệ thống.** Arp và cộng sự (USENIX Security 2022) khảo sát 30 bài top-tier trong 10 năm và chỉ ra các "pitfall" lặp đi lặp lại — trong đó *data snooping*, *spurious correlations*, và *inappropriate baselines* trực tiếp tạo ra hiệu năng thổi phồng. Phiên bản tạp chí (CACM 2024, "Pitfalls in Machine Learning for Computer Security") khẳng định các lỗi này vẫn phổ biến.

2. **Bộ dữ liệu nền tảng có khiếm khuyết đo được.** Engelen và cộng sự (WTMC/IEEE SPW 2021) tái dựng CICIDS2017, phát hiện lỗi sắp xếp/trùng lặp gói, tấn công bị gán nhãn sai, và lỗi trong CICFlowMeter; họ phải **gán nhãn lại > 20%** số luồng. Lanvin và cộng sự (2023) định lượng rằng các lỗi này tạo ra **khác biệt hiệu năng có ý nghĩa thống kê**. Dube (2024) còn lập luận mạnh hơn: mô hình huấn luyện trên dữ liệu **dạng tóm tắt luồng** của CIC-IDS 2017 khó có giá trị thực tiễn.

3. **Hiệu năng cao trên dữ liệu tổng hợp không tổng quát.** Cantone và cộng sự (IEEE Access 2024) cho thấy mô hình đạt gần hoàn hảo khi train/test cùng bộ, nhưng **rơi về mức ngẫu nhiên** khi train trên bộ này và test trên bộ khác. Hesford và cộng sự ("Expectations Versus Reality", 2024) xác nhận: không IDS nào thắng tuyệt đối, hiệu năng phụ thuộc mạnh vào loại tấn công và môi trường.

Khi đặt cạnh nhau, các bằng chứng này cho thấy **đánh giá đáng tin là điều kiện tiên quyết** cho mọi tuyên bố tiến bộ trong NIDS học máy — và là một ngách còn ít người làm nghiêm túc, đặc biệt khi gắn với ràng buộc triển khai biên.

---

## 2. Phương pháp tổng quan (giao thức kiểu PRISMA rút gọn)

Tổng quan này theo một giao thức kiểu PRISMA rút gọn, mô tả trung thực những gì thực sự tra cứu được (không phải một SLR đầy đủ với hàng nghìn bản ghi).

**Nguồn tra cứu.** Tìm kiếm web học thuật (qua công cụ WebSearch), tập trung vào: arXiv (cs.CR, cs.LG), IEEE Xplore/IEEE Access, Springer Link, ScienceDirect (Computers & Security, Engineering Applications of AI), ACM Digital Library, USENIX Security, MDPI (Future Internet, Sensors), và các trang dự án học thuật (S2Lab/UCL, DistriNet/KU Leuven).

**Từ khoá chính.** Kết hợp các cụm: *data leakage / label leakage / destination port*; *CICIDS2017 errors / troubleshooting / faulty use*; *cross-dataset generalization NIDS*; *concept drift / continual learning IDS*; *adversarial / evasion NIDS problem-space*; *federated learning IDS non-IID poisoning*; *TinyML / edge IDS quantization*; *sound evaluation malware temporal bias (TESSERACT)*; *Dos and Don'ts ML security*.

**Tiêu chí chọn (inclusion).**
- (i) Liên quan trực tiếp tới đánh giá đáng tin, rò rỉ, chất lượng dữ liệu, hoặc một trong các nhánh liền kề (drift, đối kháng, federated, biên).
- (ii) Ưu tiên 2021–2026; giữ lại một số mốc nền tảng cũ hơn (Sharafaldin 2018 — dataset gốc; TESSERACT 2019; Pierazzi 2020) vì giá trị tham chiếu.
- (iii) Có thể xác minh được tác giả/tiêu đề/nơi công bố qua tìm kiếm; có DOI hoặc URL ổn định.

**Tiêu chí loại (exclusion).**
- (i) Không xác minh được (nghi ngờ tiêu đề/tác giả/ID) → **loại bỏ** (theo nguyên tắc "không chắc thì bỏ").
- (ii) Trùng nội dung mà không bổ sung góc nhìn mới.
- (iii) Bài báo phổ thông/blog không có nội dung học thuật kiểm chứng được.

**Quy mô thực tế.** Khoảng 16–18 truy vấn tìm kiếm được thực hiện; sau khử trùng và lọc xác minh, **35 nguồn** được giữ lại trong danh mục tham khảo (mục 7), trong đó ~28 nguồn thuộc 2021–2026. Một số ID arXiv "tương lai" (vd 2602.*, 2603.*) xuất hiện trong kết quả tìm kiếm nhưng **không được đưa vào** danh mục chính trừ khi có trang công bố ổn định, để giữ độ tin cậy.

**Hạn chế của giao thức.** Đây là tổng quan có định hướng (trục chính cố định trước), không phải khảo sát toàn diện; có thiên lệch về tài liệu tiếng Anh và nguồn có chỉ mục web. Mục tiêu là **độ sâu theo trục** và **độ tin của từng nguồn**, không phải độ phủ tối đa.

---

## 3. Phân loại có cấu trúc (taxonomy)

Taxonomy được tổ chức theo sáu nhánh. Nhánh (a) và (b) là **lõi** của trục chính; (c)–(f) là các điều kiện cần để một đánh giá là đáng tin trong thực tế.

### (a) Nguồn rò rỉ dữ liệu (data leakage)

Rò rỉ là khi thông tin lẽ ra không có ở thời điểm suy luận lại "rò" vào quá trình huấn luyện/đánh giá, làm hiệu năng báo cáo cao giả tạo. Bốn dạng chính:

- **(a1) Rò rỉ nhãn qua đặc trưng (label leakage / feature leakage).** Đặc trưng tương quan nhân tạo với nhãn. Ví dụ kinh điển trên CICIDS2017 là `destination_port`: trong testbed, mỗi loại tấn công thường gắn với một cổng cố định, nên mô hình "học cổng" thay vì học hành vi tấn công (UNB CIC; thảo luận trong nhiều khảo sát dataset). Đây chính là đặc trưng mà quy trình của NCS đã chủ động loại bỏ.
- **(a2) Trùng lặp và near-duplicate.** Cùng một luồng (hoặc luồng gần trùng ở mức đặc trưng) xuất hiện ở cả tập train và test → "ghi nhớ" thay vì khái quát hoá. Khử trùng lặp ở mức đặc trưng là một bước loại-trừ-rò-rỉ trực tiếp.
- **(a3) Rò rỉ thời gian (temporal leakage).** Chia train/test ngẫu nhiên khiến mẫu "tương lai" lọt vào tập huấn luyện, tạo cấu hình bất khả thi trong thực tế. TESSERACT (Pendlebury và cộng sự, USENIX 2019) hình thức hoá *temporal bias* và *spatial bias*, đề xuất ràng buộc thời gian/không gian để loại bỏ — khung tham chiếu nền tảng cho đánh giá theo thời gian (`IDS_SPLIT_MODE=temporal` của NCS là hiện thực hoá đúng tinh thần này).
- **(a4) Rò rỉ qua chọn lọc mô hình (model-selection leakage).** Dùng tập test để chọn mô hình/đặc trưng/siêu tham số (data snooping). Arp và cộng sự liệt kê đây là pitfall phổ biến. Cách phòng: chọn mô hình thành phần trên **tập validation tách riêng**, giữ test hoàn toàn ngoài khâu chọn (đúng như thiết kế ensemble Top-3 của NCS).

### (b) Khiếm khuyết bộ dữ liệu và độ thực tế (dataset defects & realism)

- **(b1) Lỗi nhãn và lỗi trích đặc trưng.** Engelen và cộng sự (2021): lỗi CICFlowMeter, tấn công không gán nhãn, gói trùng/sai thứ tự; gán nhãn lại > 20% luồng. Lanvin và cộng sự (2023): các lỗi này thay đổi xếp hạng hiệu năng một cách có ý nghĩa.
- **(b2) Độ thực tế của biểu diễn.** Dube (2024): dữ liệu tóm tắt luồng đánh mất thông tin payload/raw, hạn chế giá trị thực tiễn; khuyến nghị dùng PCAP/raw.
- **(b3) Khoảng cách phân phối và độ đa dạng.** Goldschmidt & Chudá (Computers & Security 2025) khảo sát hệ thống **89 bộ dữ liệu NIDS** trên 13 thuộc tính, kết luận cộng đồng thiếu chuẩn lưu trữ/thực hành, dẫn tới đánh giá thiên lệch và kết quả "quá lạc quan". Bộ dữ liệu hiện đại hơn: CIC-IoT-2023 (Neto và cộng sự) với 105 thiết bị, 33 tấn công — tăng độ thực tế cho miền IoT.

### (c) Giao thức đánh giá (evaluation protocol)

- **(c1) Đánh giá per-class.** Accuracy/F1 tổng thể che giấu thất bại ở lớp tấn công hiếm (mất cân bằng lớp). Cần báo cáo precision/recall/F1 theo từng lớp.
- **(c2) Kiểm định thống kê.** Khác biệt giữa mô hình phải qua kiểm định ý nghĩa (vd permutation test, McNemar) để loại trừ ngẫu nhiên — đúng quy trình NCS áp dụng.
- **(c3) Khái quát hoá liên bộ dữ liệu (cross-dataset).** Train một bộ, test bộ khác. Cantone và cộng sự (2024) trên CIC-IDS-2017/CSE-CIC-IDS2018/LycoS cho thấy sụp đổ hiệu năng → đây là phép thử khái quát hoá thật.
- **(c4) Robustness và độ tin thực hành.** Hesford và cộng sự (2024): so sánh IDS thực tế (HELAD, AOC-IDS, NEGSC, SLIPS) trên 5 bộ; nhấn mạnh khó khăn tái lập từ code/repo.

### (d) Thích ứng trôi khái niệm (drift / continual learning)

- **(d1) Phát hiện trôi.** Mối đe doạ và hành vi bình thường biến đổi → mô hình tĩnh suy giảm. Khảo sát của Agrawal và cộng sự (Engineering Applications of AI 2024) tổng hợp drift và feature dynamics trong IDS (2019–2024).
- **(d2) Học liên tục, cân bằng quên–học.** SSF (Han và cộng sự, 2024) — chọn lọc và quên có chiến lược với memory buffer; CITADEL (2025) — phát hiện bất thường liên tục cho IoT IDS. Khảo sát hợp nhất drift/forgetting/adaptation (arXiv 2505.17902, 2025) cho khung lý thuyết stability–plasticity.

### (e) Bền vững đối kháng (adversarial robustness)

- **(e1) Evasion trong không-gian-đặc-trưng vs không-gian-vấn-đề.** Pierazzi và cộng sự (IEEE S&P 2020) hình thức hoá *problem-space attack* với ràng buộc khả thi (giữ ngữ nghĩa, bền với tiền xử lý) — chìa khoá để phân biệt "perturbation toán học" với "lưu lượng tấn công thật".
- **(e2) Tính khả thi thực tế của evasion với NIDS.** Sheatsley và cộng sự / nhóm tác giả của arXiv 2306.05494 (2023–2024) lập luận evasion với NIDS thường **bất khả thi trong thực tế**, đặc biệt với hệ động (dynamic), do khó truy cập vector đặc trưng và ràng buộc miền mạng.
- **(e3) Tổng quan và phòng thủ.** "A Review of the Duality of Adversarial Learning in Network Intrusion" (arXiv 2412.13880, 2024); khung phòng thủ chống tấn công đối kháng cho ML-NIDS (arXiv 2502.15561, 2025).

### (f) Ràng buộc biên và hiệu năng (edge / TinyML)

- **(f1) Đồng thiết kế mô hình–phần cứng.** TinyML, quantization, pruning, ONNX/TensorRT để suy luận trên MCU/ARM; đánh đổi năng lượng–độ trễ–chính xác–robustness.
- **(f2) Bằng chứng gần đây.** Khảo sát/nghiên cứu 2025–2026: TinyML cho IDS tiết kiệm năng lượng trên thiết bị ràng buộc (jisis.org 2025; ScienceDirect 2025); IDS nhẹ, ý thức năng lượng cho IIoT dùng TinyML và Edge AI (Scientific Reports 2026); TinyML đa phân loại bảo vệ riêng tư cho edge-IoT (Computing/Springer 2025).
- **(f3) Liên kết phân tán bảo mật (federated, liền kề).** FL cho IDS dưới non-IID và đầu độc (poisoning): P4P (ScienceDirect 2026), khảo sát PPFL-IDS-IoT (ACM Computing Surveys 2025). Quan trọng cho NCS: *rò rỉ trong bối cảnh phân tán còn ít được kiểm soát*.

### (g) Benchmark hướng đối thủ thích nghi (cross-cutting)

- StealthCup (arXiv 2511.17761, ACM AsiaCCS 2025): biến benchmark IDS thành CTF đa tầng, hướng evasion, với pentester thật trên testbed IT/OT — nắm bắt **hành vi đối thủ thích nghi** mà dataset tĩnh (NSL-KDD, CICIDS2017) bỏ lỡ. Đây là hình mẫu cho "đối thủ thích nghi" trong chuẩn đánh giá của NCS.

---

## 4. Bảng tổng hợp công trình tiêu biểu

| # | Công trình (tác giả, năm) | Trọng tâm | Bộ dữ liệu | Xử lý rò rỉ? | Hạn chế chính |
|---|---|---|---|---|---|
| 1 | Sharafaldin và cộng sự, 2018 | Giới thiệu CICIDS2017 | CICIDS2017 (gốc) | Không (chính là nguồn lỗi sau này) | Lỗi nhãn/trích đặc trưng, `dst_port` rò rỉ |
| 2 | Pendlebury và cộng sự (TESSERACT), 2019 | Loại thiên lệch không-gian/thời-gian | Android malware | Có (temporal/spatial bias) | Miền malware, chưa NIDS |
| 3 | Pierazzi và cộng sự, 2020 | Adversarial problem-space | Android malware | Gián tiếp (tính khả thi) | Không phải miền mạng |
| 4 | Engelen và cộng sự, 2021 | Sửa lỗi & gán nhãn lại CICIDS2017 | CICIDS2017 | Có (label/feature errors) | Chỉ một bộ; cần dùng CICFlowMeter sửa |
| 5 | Arp và cộng sự, 2022 (CACM 2024) | 10 pitfall ML trong an ninh | Khảo sát 30 bài | Có (data snooping, spurious) | Tổng quát, không đưa benchmark |
| 6 | Lanvin và cộng sự, 2023 | Định lượng tác động lỗi nhãn | CICIDS2017 (sửa) | Có | Phạm vi một bộ |
| 7 | Cantone và cộng sự, 2024 | Cross-dataset generalization | CIC-IDS-2017/2018, LycoS | Có (phát hiện anomaly dữ liệu) | Bốn bộ cùng họ CIC |
| 8 | Dube, 2024 | Phê phán dữ liệu tóm tắt luồng | CIC-IDS 2017 | Có (lập luận realism) | Định tính, ít thực nghiệm |
| 9 | Hesford và cộng sự, 2024 | Đánh giá IDS thực tế | 5 bộ (CICIDS2017, UNSW...) | Một phần | Khó tái lập từ repo |
| 10 | Agrawal và cộng sự, 2024 | Khảo sát concept/feature drift IDS | Nhiều | Gián tiếp (temporal) | Khảo sát, ít thực nghiệm mới |
| 11 | Han và cộng sự (SSF), 2024 | Continual learning IDS | NSL-KDD, UNSW-NB15 | Gián tiếp | Chưa ràng buộc biên |
| 12 | Goldschmidt & Chudá, 2025 | Khảo sát 89 bộ dữ liệu NIDS | 89 bộ | Có (chỉ ra over-optimism) | Không sửa dữ liệu, chỉ khuyến nghị |
| 13 | CITADEL, 2025 | Continual anomaly detection IoT IDS | IoT | Gián tiếp | Mới, ít đối sánh |
| 14 | "Duality of Adversarial Learning", 2024 | Tổng quan tấn công/phòng thủ NIDS | Nhiều | Không trực tiếp | Khảo sát |
| 15 | arXiv 2306.05494, 2023–24 | Tính bất khả thi evasion với NIDS | NIDS | Gián tiếp (đe doạ) | Tranh luận, chưa dứt điểm |
| 16 | StealthCup, 2025 | Benchmark CTF hướng evasion | Testbed IT/OT mới | N/A (đánh giá thật) | Tốn công, khó nhân rộng |
| 17 | Neto và cộng sự (CIC-IoT-2023), 2023 | Bộ dữ liệu IoT quy mô lớn | CIC-IoT-2023 | Không (dataset mới) | Cần kiểm toán rò rỉ độc lập |
| 18 | TinyML IIoT IDS, 2025–2026 | IDS nhẹ, năng lượng-aware | IoT/IIoT | Hiếm khi | Ít đánh giá robustness/drift |

*Ghi chú: "Xử lý rò rỉ?" đánh giá mức công trình **trực tiếp** giải quyết rò rỉ/đánh giá đáng tin, không phải chỉ nhắc tới.*

---

## 5. Khoảng trống và cơ hội nghiên cứu

Nối thẳng vào bốn trục của Kịch bản 1 (A — học liên tục; C — đối kháng; F — đánh giá & dữ liệu đáng tin; G — hiệu quả biên), với Trục F làm sợi chỉ đỏ.

**Khoảng trống G1 — Thiếu công cụ kiểm toán rò rỉ tổng quát, tự động.** Hầu hết công trình xử lý rò rỉ theo kiểu thủ công, ad-hoc cho từng bộ (loại `dst_port`, khử trùng lặp). Chưa có một **bộ công cụ kiểm toán rò rỉ tự động** phát hiện đồng thời label leakage, near-duplicate, và temporal leakage trên nhiều bộ.
→ *Ngách NCS (Trục F):* tổng quát hoá quy trình loại-trừ-rò-rỉ của luận văn thành công cụ kiểm toán tái lập được, áp dụng cho CICIDS2017, CSE-CIC-IDS2018, CIC-IoT-2023.

**Khoảng trống G2 — Đánh giá đáng tin chưa gắn với đối thủ thích nghi và miền mạng thật.** Dataset tĩnh không nắm bắt đối thủ thích nghi (StealthCup chỉ ra rõ); tranh luận evasion (arXiv 2306.05494) cho thấy ràng buộc khả thi miền mạng chưa được tích hợp vào benchmark chuẩn.
→ *Ngách NCS (Trục C + F):* một chuẩn đánh giá **loại-trừ-rò-rỉ + đối thủ thích nghi**, kiểm tra evasion trong không-gian-vấn-đề (giữ chức năng tấn công), đo cả gate lẫn classifier của pipeline hai tầng.

**Khoảng trống G3 — Thích ứng trôi khái niệm chưa đặt dưới ràng buộc biên.** SSF, CITADEL... tập trung server; ít công trình đánh giá continual learning **dưới ngân sách tài nguyên ARM/Jetson** với đánh giá theo thời gian không rò rỉ.
→ *Ngách NCS (Trục A + G):* học liên tục rẻ (drift detection + replay chọn lọc + distillation) chạy được trên Jetson, đánh giá theo kịch bản trôi thật (temporal split), không quên thảm khốc.

**Khoảng trống G4 — Đánh đổi đa mục tiêu năng lượng–chính xác–robustness chưa tối ưu đồng thời.** TinyML IDS tối ưu năng lượng/độ trễ nhưng hiếm khi đo robustness đối kháng hay khả năng thích ứng drift cùng lúc.
→ *Ngách NCS (Trục G + C):* đồng thiết kế mô hình–phần cứng (quantization/TensorRT) **không đánh mất robustness**, tối ưu Pareto bốn chiều, đo trên cụm Jetson thật với tegrastats.

**Khoảng trống G5 — Rò rỉ trong bối cảnh phân tán/liên kết chưa được kiểm soát.** FL-IDS giải quyết non-IID/poisoning nhưng đánh giá loại-trừ-rò-rỉ trong môi trường phân tán gần như vắng mặt.
→ *Cơ hội mở rộng (liền kề Kịch bản 2):* mở rộng kiểm toán rò rỉ sang FL.

**Định vị tổng thể của NCS.** Bốn tài sản của luận văn — quy trình loại-trừ-rò-rỉ + kiểm định thống kê, hạ tầng phân tán/biên thật (Spark + Jetson), pipeline streaming, và XAI/so-sánh-engine — tạo một **testbed tự nhiên** để biến đánh giá đáng tin thành đóng góp khoa học chính, thay vì chỉ "thêm một mô hình DL".

---

## 6. Kết luận

Tài liệu 2018–2026 hội tụ về một thông điệp nhất quán: **vấn đề lớn nhất của NIDS học máy không phải mô hình, mà là đánh giá**. Rò rỉ dữ liệu (label/temporal/duplicate/model-selection), khiếm khuyết bộ dữ liệu (lỗi nhãn CICIDS2017, dữ liệu tóm tắt mất thông tin), và giả định phân phối tĩnh cùng tạo ra hiệu năng "gần hoàn hảo" không chuyển sang thực tế. Các công trình nền tảng (Arp và cộng sự; TESSERACT; Engelen/Lanvin; Goldschmidt & Chudá; Cantone và cộng sự) đã chẩn đoán rõ căn bệnh, nhưng **giải pháp vẫn rời rạc**: chưa có một chuẩn đánh giá đáng tin, tự động kiểm toán rò rỉ, có đối thủ thích nghi, tái lập được, và **khả thi trên phần cứng biên**.

Đây chính là ngách của NCS theo Kịch bản 1: lấy quy trình loại-trừ-rò-rỉ + kiểm định thống kê + cross-dataset của luận văn làm hạt nhân (Trục F), bện cùng thích ứng trôi khái niệm (A), bền vững đối kháng (C), và đồng thiết kế biên (G), trên hạ tầng Spark + Jetson sẵn có. Đóng góp kỳ vọng là một **khung đánh giá đáng tin cho IDS biên** — chữ ký học thuật khác biệt, ít người làm nghiêm túc, và có lợi thế cạnh tranh rõ ràng.

---

## 7. Tài liệu tham khảo

*Tất cả nguồn dưới đây đã được tra cứu/xác minh qua tìm kiếm web (tiêu đề, tác giả, nơi công bố). Ưu tiên 2021–2026.*

**Rò rỉ dữ liệu, pitfall, đánh giá đáng tin**

1. Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., Cavallaro, L., Rieck, K. (2022). *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security Symposium 2022, pp. 3971–3988. https://www.usenix.org/conference/usenixsecurity22/presentation/arp
2. Arp, D. et al. (2024). *Pitfalls in Machine Learning for Computer Security.* Communications of the ACM. https://dl.acm.org/doi/10.1145/3643456
3. Pendlebury, F., Pierazzi, F., Jordaney, R., Kinder, J., Cavallaro, L. (2019). *TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time.* USENIX Security 2019. https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury (Extended: https://arxiv.org/abs/2402.01359)

**CICIDS2017: lỗi, sửa, độ thực tế**

4. Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018, pp. 108–116. https://www.scitepress.org/papers/2018/66398/66398.pdf
5. Engelen, G., Rimmer, V., Joosen, W. (2021). *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.* IEEE Security and Privacy Workshops (WTMC 2021). https://intrusion-detection.distrinet-research.be/WTMC2021/Resources/wtmc2021_Engelen_Troubleshooting.pdf (Code: https://github.com/GintsEngelen/WTMC2021-Code)
6. Lanvin, M. et al. (2023). *Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes.* Springer (CRiSIS 2022), LNCS. https://link.springer.com/chapter/10.1007/978-3-031-31108-6_2
7. Dube, R. (2024). *Faulty use of the CIC-IDS 2017 dataset in information security research.* Journal of Computer Virology and Hacking Techniques. https://link.springer.com/article/10.1007/s11416-023-00509-7
8. CIC, University of New Brunswick. *Intrusion Detection Evaluation Dataset (CIC-IDS2017).* https://www.unb.ca/cic/datasets/ids-2017.html

**Chất lượng dữ liệu, khái quát hoá, đánh giá thực tế**

9. Goldschmidt, P., Chudá, D. (2025). *Network Intrusion Datasets: A Survey, Limitations, and Recommendations.* Computers & Security, vol. 156. https://arxiv.org/abs/2502.06688 (DOI: https://doi.org/10.1016/j.cose.2025.104510 ; Repo: https://github.com/xGoldy/nid-datasets)
10. Cantone, M., Marrocco, C., Bria, A. (2024). *Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study.* IEEE Access, vol. 12, pp. 144489–144508. https://arxiv.org/abs/2402.10974
11. Hesford, J., Cheng, D., Wan, A., Huynh, L., Kim, S., Kim, H., Hong, J. B. (2024). *Expectations Versus Reality: Evaluating Intrusion Detection Systems in Practice.* arXiv:2403.17458 (IEEE DSN-S 2025). https://arxiv.org/abs/2403.17458
12. *Cross-Dataset Temporal and Semantic Generalization of Intrusion Detection Models for the Future Internet.* (2025–2026). MDPI Future Internet, 18(4):194. https://www.mdpi.com/1999-5903/18/4/194

**Thích ứng trôi khái niệm / học liên tục**

13. Agrawal, A. et al. (2024). *Evolving cybersecurity frontiers: A comprehensive survey on concept drift and feature dynamics aware machine and deep learning in intrusion detection systems.* Engineering Applications of Artificial Intelligence, vol. 137. https://www.sciencedirect.com/science/article/pii/S0952197624013010
14. Han, X. et al. (2024). *Continual Learning with Strategic Selection and Forgetting for Network Intrusion Detection (SSF).* arXiv:2412.16264. https://arxiv.org/abs/2412.16264
15. *CITADEL: Continual Anomaly Detection for Enhanced Learning in IoT Intrusion Detection.* (2025). arXiv:2508.19450. https://arxiv.org/pdf/2508.19450
16. *Evolving Machine Learning in Non-Stationary Environments: A Unified Survey of Drift, Forgetting, and Adaptation.* (2025). arXiv:2505.17902. https://arxiv.org/pdf/2505.17902
17. *Continual Learning for IDS Under Evolving Network Threats.* (2025). MDPI Future Internet, 17(10):456. https://www.mdpi.com/1999-5903/17/10/456

**Bền vững đối kháng / evasion**

18. Pierazzi, F., Pendlebury, F., Cortellazzi, J., Cavallaro, L. (2020). *Intriguing Properties of Adversarial ML Attacks in the Problem Space.* IEEE Symposium on Security and Privacy, pp. 1332–1349. DOI: 10.1109/SP40000.2020.00073. https://arxiv.org/pdf/1911.02142
19. *Evasion Adversarial Attacks Remain Impractical Against ML-based Network Intrusion Detection Systems, Especially Dynamic Ones.* (2023–2024). arXiv:2306.05494. https://arxiv.org/abs/2306.05494
20. *A Review of the Duality of Adversarial Learning in Network Intrusion: Attacks and Countermeasures.* (2024). arXiv:2412.13880. https://arxiv.org/pdf/2412.13880
21. *A Defensive Framework Against Adversarial Attacks on Machine Learning-Based Network Intrusion Detection Systems.* (2025). arXiv:2502.15561. https://arxiv.org/html/2502.15561v1
22. *Adversarial Challenges in Network Intrusion Detection Systems: Research Insights and Future Prospects.* (2024). arXiv:2409.18736. https://arxiv.org/pdf/2409.18736

**Benchmark hướng đối thủ thích nghi**

23. *StealthCup: Realistic, Multi-Stage, Evasion-Focused CTF for Benchmarking IDS.* (2025). arXiv:2511.17761 (ACM AsiaCCS). https://arxiv.org/abs/2511.17761 (DOI: https://doi.org/10.1145/3779208.3806088 ; Repo: https://github.com/ait-cs-IaaS/StealthCup2025)

**Bộ dữ liệu IoT hiện đại**

24. Neto, E. C. P. et al. (2023). *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment.* Sensors (PMC). https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/ (Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html)

**Học liên kết phân tán (federated) cho IDS**

25. *P4P: A probe-guided anti-poisoning defense for federated learning-based intrusion detection in IoT networks under non-IID data.* (2026). Journal of Network and Computer Applications, ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S1084804526000779
26. Bai, X. et al. (2025). *Enhancing IoT Security via Federated Learning: A Comprehensive Approach to Intrusion Detection.* IET Information Security. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ise2/8432654
27. *Privacy-Preserving Federated Learning for Intrusion Detection in IoT Environments: A Survey.* (2024–2025). https://www.researchgate.net/publication/383771026
28. *Federated Learning-Based Intrusion Detection in IoT Networks: Performance Evaluation and Data Scaling Study.* (2025). MDPI JSAN, 14(4):78. https://www.mdpi.com/2224-2708/14/4/78

**Biên / TinyML**

29. *TinyML-based intrusion detection systems for sustainable and energy-constrained IoT devices.* (2025). ScienceDirect. https://www.sciencedirect.com/science/article/pii/S2590123025040642
30. *Lightweight and Energy-Aware Intrusion Detection for Industrial IoT Using TinyML and Edge AI.* (2026). Scientific Reports. https://www.nature.com/articles/s41598-026-50690-0
31. *TinyML strategies for privacy-preserving and cyber threat multi-classification in edge-IoT networks.* (2025). Computing (Springer). https://link.springer.com/article/10.1007/s00607-025-01522-y
32. *Tiny ML-Enabled Energy-Efficient Intrusion Detection.* (2025). JISIS. https://jisis.org/wp-content/uploads/2025/11/2025.I3.041.pdf

**Tham chiếu phương pháp luận liên ngành (malware, dùng làm khung đánh giá)**

33. Pendlebury, F. et al. — Trang dự án TESSERACT (S2Lab/UCL). https://s2lab.cs.ucl.ac.uk/projects/tesseract/
34. Engelen, G. — CICFlowMeter (bản sửa lỗi). https://github.com/GintsEngelen/CICFlowMeter
35. *Deep Learning-based Intrusion Detection: A Survey.* (2025). arXiv:2504.07839. https://arxiv.org/abs/2504.07839

---

*Lưu ý xác minh: Các nguồn có ID arXiv dạng "tương lai" (2602.*, 2603.*) xuất hiện trong kết quả tìm kiếm đã được **loại khỏi danh mục chính** do chưa có trang công bố ổn định tại thời điểm soạn, nhằm bảo đảm không đưa nguồn không kiểm chứng được. Một số nguồn đã có sẵn trong HUONG_PHAT_TRIEN_NCS.md được giữ lại và đối chiếu lại để bảo đảm tính nhất quán.*
