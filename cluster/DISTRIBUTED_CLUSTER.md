# Triển khai phân tán: 1× Mac + 2× Jetson Nano

Toàn bộ pipeline (huấn luyện ML **và** inference edge) dùng **Spark standalone cluster** 3 node.

```
┌─────────────────────────────────────────────────────────────┐
│  Mac                                                         │
│  • Spark Master (:7077) + Web UI (:8080)                    │
│  • Docker: Kafka, PostgreSQL, InfluxDB, Grafana             │
│  • ml_00 (CSV → parquet), save_model.py                     │
└───────────────┬─────────────────────┬───────────────────────┘
                │                     │
     ┌──────────┴──────────┐ ┌────────┴──────────┐
     │  Jetson Nano #1       │ │  Jetson Nano #2   │
     │  Spark Worker         │ │  Spark Worker     │
     │  PySpark Driver (ML)  │ │  edge classifier  │
     │  anomaly_gate (opt.)  │ │  (SOICT mode A)   │
     └───────────────────────┘ └───────────────────┘
```

## Bước 0 — Cấu hình một lần

```bash
cd /path/to/Thesis_IDS
cp cluster/spark_cluster.env.example cluster/spark_cluster.env
# Sửa MAC_IP, JETSON1_IP, JETSON2_IP, SSH user, đường dẫn CSV
```

Trên **mỗi Jetson** (một lần):

```bash
cd ~/Thesis_IDS/raspberry && ./scripts/setup_jetson.sh
```

## Bước 1 — Khởi động cluster

**Mac:**

```bash
source cluster/spark_cluster.env
./cluster/start_master_mac.sh
cd raspberry && docker compose up -d
python scripts/init_kafka_topics.py --partitions 2
```

**Jetson #1 và #2** (SSH vào từng máy):

```bash
cd ~/Thesis_IDS
source cluster/spark_cluster.env
./cluster/start_worker.sh
```

Kiểm tra:

```bash
./cluster/check_cluster.sh
# Master UI: http://<MAC_IP>:8080 — phải thấy 2 workers ALIVE
```

## Bước 2 — Đồng bộ code + data

Parquet phải tồn tại trên **cả 2 Jetson** (cùng đường dẫn):

```bash
# Mac — tạo parquet nếu chưa có
source venv/bin/activate
python ml_00_prepare_cicids2017.py

# Mac — sync sang Jetsons
source cluster/spark_cluster.env
./cluster/sync_workspace.sh
```

`IDS_CLUSTER_DATA_DIR` (mặc định `/home/jetson/Thesis_IDS/data`) phải khớp trên mọi worker.

## Bước 3 — Huấn luyện ML phân tán (FAIR / luận văn)

Driver chạy trên **Jetson #1**, executors trên **Jetson #1 + #2**, master trên **Mac**:

```bash
# Toàn bộ FAIR track
./cluster/reproduce_cluster.sh fair

# Hoặc từng script
./cluster/run_ml_remote.sh ml_01_baseline_all_features.py
./cluster/run_ml_remote.sh ml_07_cross_method_comparison.py
```

SOICT / edge prep:

```bash
./cluster/reproduce_cluster.sh soict
```

Luận văn đầy đủ:

```bash
./cluster/reproduce_cluster.sh thesis
```

## Bước 4 — Edge inference phân tán (SOICT)

Cập nhật `raspberry/.env` trên Jetsons:

```env
SPARK_MASTER=spark://<MAC_IP>:7077
KAFKA_BOOTSTRAP_SERVERS=<MAC_IP>:9092
```

**Mode A** (khuyến nghị): Jetson1 `anomaly_gate`, Jetson2 `classifier` — xem `raspberry/JETSON_DISTRIBUTED.md`.

Spark classifier trên Jetson #2 dùng cluster (executors trên cả 2 Jetson).

## Biến môi trường quan trọng

| Biến | Vai trò |
|------|---------|
| `IDS_SPARK_CLUSTER=1` | Bật chế độ cluster trong `shared_utils.py` |
| `SPARK_MASTER` | `spark://<MAC_IP>:7077` |
| `SPARK_DRIVER_HOST` | IP Jetson #1 (nơi chạy driver Python) |
| `IDS_CLUSTER_DATA_DIR` | Parquet path trên mọi worker |
| `SPARK_EXECUTOR_MEMORY` | `768m` (Jetson 4GB) |

## Bắt buộc phân tán

**Không còn chế độ local Spark** cho `ml_01`–`ml_07`. Nếu chạy trực tiếp:

```bash
source cluster/load_cluster_env.sh   # hoặc dùng reproduce_cluster.sh
./cluster/run_ml_remote.sh ml_01_baseline_all_features.py
```

Chỉ `ml_00` và `save_model.py` trên Mac được phép local (`IDS_ALLOW_LOCAL_SPARK=1`).

## Troubleshooting

| Lỗi | Cách xử lý |
|-----|------------|
| Worker không xuất hiện trên UI | Firewall Mac port 7077; đúng `MAC_IP` |
| FileNotFound parquet trên worker | Chạy `./cluster/sync_workspace.sh` |
| OOM trên Jetson | Giảm `SPARK_EXECUTOR_MEMORY`, `EDGE_BATCH_SIZE` |
| Driver timeout | Tăng `spark.network.timeout` (đã set 800s) |
| SHAP chậm | SHAP chạy trên driver (Jetson1) — bình thường |

## Kiến trúc code

- `shared_utils.py` — `create_spark_session()` đọc `SPARK_MASTER`, `IDS_SPARK_CLUSTER`
- `cluster/run_ml_remote.sh` — SSH driver trên Jetson1
- `cluster/sync_workspace.sh` — rsync Mac → 2 Jetsons
