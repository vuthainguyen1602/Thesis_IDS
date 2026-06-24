# Triển khai phân tán IDS trên 2 Jetson Orin Nano Super Developer Kit (8GB)

Hướng dẫn chạy hệ thống IDS edge trên **2 Jetson Orin Nano Super Developer Kit (8GB RAM, 256GB SSD)**, kết nối với hạ tầng Kafka/PostgreSQL/InfluxDB trên Mac.

**Huấn luyện ML phân tán (Spark):** xem [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md)

---

## IP lab (tham chiếu)

| Node | IP | Ghi chú |
|------|-----|---------|
| Mac | `192.168.1.165` | Spark Master, Docker — `MAC_IP` |
| Jetson #1 | `192.168.1.50` | Driver + Worker, `anomaly_gate` |
| Jetson #2 | `192.168.1.205` | Worker, `classifier` |

Kiểm tra IP Mac: `ipconfig getifaddr en0` — phải khớp `MAC_IP` trong `cluster/spark_cluster.env` và `KAFKA_ADVERTISED_LISTENERS` trong `docker-compose.yml`.

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                         PC / Mac                                 │
│  Docker: Kafka, PostgreSQL, InfluxDB, Grafana                   │
│  data_sender.py → topic: ids-network-flow                       │
└───────────────┬───────────────────────────────┬─────────────────┘
                │                               │
     ┌──────────┴──────────┐         ┌──────────┴──────────┐
     │   Jetson Orin Nano Super #1    │         │   Jetson Orin Nano Super #2    │
     │   (anomaly_gate)    │────────▶│   (classifier)      │
     │   sklearn AE filter │  Kafka  │   PySpark model     │
     └─────────────────────┘         └─────────────────────┘
              ids-suspicious-flow
```

Hệ thống hỗ trợ **3 chế độ phân tán**:

| Chế độ | Mô tả | Khi nào dùng |
|--------|-------|--------------|
| **A. Pipeline split** | Jetson #1 = anomaly gate, Jetson #2 = classifier | Giảm tải Spark, phù hợp luận văn |
| **B. Horizontal scaling** | Cả 2 Jetson chạy pipeline đầy đủ, cùng consumer group | Tăng throughput |
| **C. Spark cluster** | Jetson #1 = Spark master, Jetson #2 = worker | Phân tán inference Spark |

---

## Yêu cầu phần cứng & mạng

- 2× Jetson Orin Nano Super Developer Kit (**8GB RAM**, **256GB SSD** khuyến nghị)
- Mac/PC cùng mạng LAN WiFi với 2 Jetson (`192.168.1.x`)
- Nguồn 5V/4A ổn định cho mỗi Jetson
- Swap 4GB (tùy chọn trên 8GB — `setup_jetson.sh` tự cấu hình)

---

## Bước 1: Hạ tầng trên PC/Mac

**Spark cluster (huấn luyện ML phân tán):** xem [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md)

```bash
cd raspberry/
docker compose up -d

# Tạo Kafka topics với ≥ 2 partitions
source venv/bin/activate   # hoặc pip install kafka-python
python scripts/init_kafka_topics.py --partitions 2
```

Cập nhật IP Mac (`192.168.1.165`) trong `docker-compose.yml` (dòng `KAFKA_ADVERTISED_LISTENERS`).

Xuất model (nếu chưa có):

```bash
python scripts/save_model.py
# Tuỳ chọn anomaly gate:
cd .. && python ml_08_anomaly_gate_autoencoder.py
```

---

## Bước 2: Setup từng Jetson Orin Nano Super

Trên **cả 2 Jetson**:

```bash
scp -r raspberry/ jetson@<jetson-ip>:~/raspberry
scp -r raspberry/model/* jetson@<jetson-ip>:~/raspberry/model/

ssh jetson@<jetson-ip>
cd ~/raspberry
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

---

## Bước 3: Chọn chế độ triển khai

### Chế độ A — Pipeline split (khuyến nghị)

**Jetson #1** — anomaly gate (lọc traffic nhẹ, không cần Spark):

```bash
cp .env.jetson1.example .env
nano .env   # sửa IP PC/Mac
source venv/bin/activate
python edge/kafka_consumer.py
```

**Jetson #2** — Spark classifier:

```bash
cp .env.jetson2.example .env
nano .env   # sửa IP PC/Mac
source venv/bin/activate
python edge/kafka_consumer.py
```

Biến môi trường quan trọng:

| Jetson | EDGE_NODE_ID | EDGE_NODE_ROLE | ALERT_ENABLED |
|--------|--------------|----------------|---------------|
| #1 | `jetson-nano-1` | `anomaly_gate` | `0` |
| #2 | `jetson-nano-2` | `classifier` | `1` |

Luồng dữ liệu:
1. PC gửi flows → `ids-network-flow`
2. Jetson #1 chấm điểm autoencoder, forward flows nghi ngờ → `ids-suspicious-flow`
3. Jetson #2 classify bằng PySpark, lưu kết quả + gửi alert

---

### Chế độ B — Horizontal scaling

Cả 2 Jetson chạy pipeline đầy đủ, **cùng** `KAFKA_GROUP_ID`:

```bash
cp .env.jetson-horizontal.example .env
# Jetson #1: EDGE_NODE_ID=jetson-nano-1
# Jetson #2: EDGE_NODE_ID=jetson-nano-2, ALERT_ENABLED=0
python edge/kafka_consumer.py
```

Kafka tự phân chia partitions cho 2 consumer → throughput gấp đôi.

---

### Chế độ C — Spark cluster (huấn luyện ML)

**Dùng Mac làm Spark Master**, 2 Jetson làm Worker. Không dùng Jetson làm master.

Xem hướng dẫn đầy đủ: [../cluster/DISTRIBUTED_CLUSTER.md](../cluster/DISTRIBUTED_CLUSTER.md)

```bash
# Mac
./cluster/start_master_mac.sh

# Mỗi Jetson
./cluster/start_worker.sh

# Mac — train
./cluster/run_ml_remote.sh ml_01_baseline_all_features.py
./cluster/pull_results.sh
```

---

## Bước 4: Gửi dữ liệu test

Trên PC/Mac:

```bash
cd raspberry/
python sender/data_sender.py --csv /path/to/dataset.csv --rate 100
```

---

## Bước 5: Giám sát

- **Grafana**: `http://<pc-ip>:3000` — metrics theo tag `host=jetson-nano-1/2`
- **PostgreSQL**: cột `node_id` trong bảng `predictions` và `alerts`
- **InfluxDB**: tag `host` = `EDGE_NODE_ID`

---

## Cấu trúc code mới

```
raspberry/edge/
├── kafka_consumer.py      # Entry point
├── role_pipelines.py      # full | anomaly_gate | classifier
├── pipeline_base.py       # Storage, monitor, alert chung
├── kafka_forwarder.py       # Forward suspicious flows
└── ...
```

Biến cấu hình mới trong `config.py`:

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `EDGE_NODE_ID` | `edge-node-1` | ID duy nhất mỗi Jetson |
| `EDGE_NODE_ROLE` | `full` | `full` / `anomaly_gate` / `classifier` |
| `KAFKA_SUSPICIOUS_TOPIC` | `ids-suspicious-flow` | Topic cho pipeline split |
| `ALERT_ENABLED` | `1` | Tắt trên node phụ để tránh alert trùng |

---

## Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| Chỉ 1 Jetson nhận message | Topic có 1 partition | `python scripts/init_kafka_topics.py --partitions 2` |
| Jetson #2 không có data | Gate chưa chạy / topic sai | Kiểm tra Jetson #1 log "Forwarded: ..." |
| PySpark OOM | RAM không đủ | Giảm `SPARK_EXECUTOR_MEMORY`; tăng swap; dùng chế độ A |
| Spark cluster không kết nối | Firewall port 7077 | `sudo ufw allow 7077` trên Jetson #1 |
| Nhiệt độ cao | Jetson không tản nhiệt | `sudo jetson_clocks`, thêm quạt |

---

## Gợi ý cho luận văn

- **Chế độ A** thể hiện rõ kiến trúc phân tán theo pipeline (edge computing)
- So sánh latency/throughput giữa 1 node vs 2 node bằng `scripts/benchmark.py`
- Grafana panel filter theo `host` tag để visualize từng Jetson
- Bảng PostgreSQL `node_id` dùng phân tích phân bổ tải
