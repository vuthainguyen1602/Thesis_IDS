# Luận văn — Intrusion Detection System

Sản phẩm **#1** trong repo: tổng hợp toàn bộ nghiên cứu (ML offline + triển khai edge phân tán).

## Quan hệ với 2 bài báo

| Thành phần | Luận văn | FAIR 2026 | SOICT 2026 |
|------------|----------|-----------|------------|
| So sánh 9 ML + ensemble | ✓ | ✓ trọng tâm | — |
| RF / SHAP / PCA | ✓ | ✓ | — |
| Drift, robustness, stats | ✓ | ✓ | — |
| Grid Search (ml_03) | ✓ | tùy chọn | — |
| Edge RPi / Jetson | ✓ | 1 đoạn | ✓ trọng tâm |
| Anomaly gate + Kafka | ✓ | — | ✓ |

## Cấu trúc

```
thesis/
├── README.md           # File này
├── reproduce.sh        # Chạy lại toàn bộ pipeline luận văn
├── collect_results.sh  # Thu figures/tables vào thesis/figures, thesis/results
├── manuscript/         # LaTeX hoặc Word luận văn
├── figures/            # Hình dùng trong luận văn (copy từ collect_results)
└── results/            # CSV, JSON, báo cáo HTML
```

## Reproduce

```bash
# Từ root repo
export IDS_ROOT="$(pwd)"
export IDS_RAW_DATA_DIR="/path/to/ids-2017"   # nếu chưa có data/

./thesis/reproduce.sh
```

Script sẽ:
1. Chuẩn bị dữ liệu (`ml_00_prepare_cicids2017.py`)
2. Chạy ml_01 → ml_07, ml_03, ml_08
3. Export model edge (`raspberry/scripts/save_model.py`)
4. Gợi ý bước edge/Jetson thủ công

Sau khi chạy xong:

```bash
./thesis/collect_results.sh
```

## Edge / Jetson (bước thủ công)

```bash
cd raspberry/
docker compose up -d
python scripts/init_kafka_topics.py --partitions 2

# Trên từng Jetson Nano — xem raspberry/JETSON_DISTRIBUTED.md
python edge/kafka_consumer.py
```

## Manuscript

Đặt file luận văn trong `manuscript/` (ví dụ `main.tex`, `references.bib`).
