# SOICT 2026 — Paper: Distributed Edge IDS on Jetson Orin Nano Super

**Hội nghị:** [SOICT 2026 — International Symposium on ICT](https://soict.org/)  
**Deadline abstract:** 09/09/2026 | **Full paper:** 16/09/2026 | **Hội nghị:** 04–05/12/2026

## Câu hỏi nghiên cứu

Kiến trúc và đánh giá hiệu năng hệ thống IDS phân tán realtime trên 2× NVIDIA Jetson Orin Nano Super với Kafka và PySpark.

## Đề xuất tiêu đề

*A Distributed Real-Time Intrusion Detection System on Jetson Orin Nano Super Edge Cluster using Kafka and PySpark*

## Code liên quan (root repo)

| Thành phần | Đường dẫn |
|------------|-----------|
| Edge pipeline | `raspberry/edge/kafka_consumer.py` |
| 3 chế độ phân tán | `raspberry/edge/role_pipelines.py` |
| Hướng dẫn Jetson | `raspberry/JETSON_DISTRIBUTED.md` |
| Anomaly gate | `ml_08_anomaly_gate_autoencoder.py` |
| Model export | `raspberry/scripts/save_model.py` |
| Benchmark | `raspberry/scripts/benchmark.py` |
| Env mẫu | `raspberry/.env.jetson1.example`, `.env.jetson2.example` |

## Reproduce

**Bước 1 — PC/Mac (train + export + infra):**

```bash
export IDS_ROOT="$(pwd)"
./papers/soict2026/reproduce.sh
```

**Bước 2 — 2× Jetson Orin Nano Super (benchmark thủ công):**

```bash
./papers/soict2026/run_benchmarks.sh   # chạy trên từng Jetson hoặc theo hướng dẫn
```

**Bước 3 — Thu kết quả:**

```bash
./papers/soict2026/collect_results.sh
```

## 3 chế độ phân tán (so sánh trong bài)

| Mode | Jetson #1 | Jetson #2 | Env |
|------|-----------|-----------|-----|
| A — Pipeline split | `EDGE_NODE_ROLE=anomaly_gate` | `EDGE_NODE_ROLE=classifier` | `.env.jetson1/2.example` |
| B — Horizontal | `EDGE_NODE_ROLE=full` | `EDGE_NODE_ROLE=full` | `.env.jetson-horizontal.example` |
| C — Spark cluster | Spark master | Spark worker | `start_spark_master/worker.sh` |

## Metrics cần báo cáo

- Throughput (flows/s)
- Latency p50 / p95 (ms)
- CPU %, RAM (MB), nhiệt độ (°C) mỗi node
- Attack detection rate
- % flows filtered by anomaly gate (mode A)

Lưu benchmark JSON vào `results/benchmarks/` trước khi chạy `collect_results.sh`.

## Manuscript

Đặt bài tiếng Anh trong `manuscript/`.

## Cross-reference

- Chọn model (DT, SHAP Top-30): trích bài FAIR hoặc `papers/fair2026/`
- Luận văn đầy đủ: `thesis/`
