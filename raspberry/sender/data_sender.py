#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Sender \u2014 Simulates Network Traffic via Kafka.\n\nReads CICIDS2017 CSV files and publishes each row as a JSON message to\nthe Kafka topic ``ids-network-flow``.  Runs on the PC/server side to\nfeed the edge IDS pipeline.

Usage::\n\n    python data_sender.py --csv /path/to/dataset.csv --rate 100

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import json
import time
import argparse
import csv

from kafka import KafkaProducer

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, DATA_CSV_PATH, SEND_RATE


def create_producer(bootstrap_servers):
    """Create a Kafka producer with JSON serialization."""
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks=1,
        retries=3,
        batch_size=16384,
        linger_ms=10,
    )
    print(f"[OK] Kafka Producer connected to {bootstrap_servers}")
    return producer


def clean_column_name(name):
    """Normalize column names: lowercase, snake_case."""
    result = name.strip().lower()
    for ch in [" ", ".", "-", "/", "(", ")"]:
        result = result.replace(ch, "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def send_csv_to_kafka(csv_path, producer, topic, rate=100):
    """
    Read a CSV file and send each row as a JSON message to Kafka.

    Args:
        csv_path: Path to the CSV file
        producer: KafkaProducer instance
        topic: Kafka topic name
        rate: Number of rows to send per second
    """
    delay = 1.0 / rate if rate > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  DATA SENDER")
    print(f"  CSV:   {csv_path}")
    print(f"  Topic: {topic}")
    print(f"  Rate:  {rate} rows/second")
    print(f"{'=' * 60}\n")

    total_sent = 0
    start_time = time.time()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Clean column names
            cleaned_fieldnames = {name: clean_column_name(name) for name in reader.fieldnames}

            for row in reader:
                # Build cleaned row
                cleaned_row = {}
                for original_name, value in row.items():
                    clean_name = cleaned_fieldnames[original_name]
                    # Try to convert numeric values
                    try:
                        if "." in value:
                            cleaned_row[clean_name] = float(value)
                        else:
                            cleaned_row[clean_name] = int(value)
                    except (ValueError, TypeError):
                        cleaned_row[clean_name] = value

                # Add timestamp
                cleaned_row["_timestamp"] = time.time()

                # Send to Kafka
                producer.send(
                    topic,
                    key=str(total_sent),
                    value=cleaned_row,
                )

                total_sent += 1

                # Rate limiting
                if delay > 0:
                    time.sleep(delay)

                # Progress logging
                if total_sent % 1000 == 0:
                    elapsed = time.time() - start_time
                    actual_rate = total_sent / elapsed if elapsed > 0 else 0
                    print(f"  Sent {total_sent:,} rows | "
                          f"Rate: {actual_rate:.1f} rows/s | "
                          f"Elapsed: {elapsed:.1f}s")

    except KeyboardInterrupt:
        print(f"\n  [WARN] Interrupted by user after {total_sent:,} rows")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"  SENDER COMPLETE")
        print(f"  Total sent:  {total_sent:,} rows")
        print(f"  Total time:  {elapsed:.1f}s")
        print(f"  Avg rate:    {total_sent / elapsed:.1f} rows/s" if elapsed > 0 else "")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="IDS Data Sender - CSV to Kafka")
    parser.add_argument("--csv", type=str, default=DATA_CSV_PATH,
                        help="Path to CSV file or directory")
    parser.add_argument("--rate", type=int, default=SEND_RATE,
                        help="Send rate (rows per second)")
    parser.add_argument("--broker", type=str, default=KAFKA_BOOTSTRAP_SERVERS,
                        help="Kafka bootstrap servers")
    parser.add_argument("--topic", type=str, default=KAFKA_TOPIC,
                        help="Kafka topic name")
    args = parser.parse_args()

    if not args.csv or not os.path.exists(args.csv):
        print(f"[ERR] CSV path not found: {args.csv}")
        print("  Set DATA_CSV_PATH in .env or pass --csv")
        sys.exit(1)

    producer = create_producer(args.broker)

    # If path is a directory, send all CSV files
    if os.path.isdir(args.csv):
        csv_files = sorted([
            os.path.join(args.csv, f)
            for f in os.listdir(args.csv)
            if f.endswith(".csv")
        ])
        print(f"  Found {len(csv_files)} CSV files in {args.csv}")
        for csv_file in csv_files:
            print(f"\n  Processing: {os.path.basename(csv_file)}")
            send_csv_to_kafka(csv_file, producer, args.topic, args.rate)
            # Recreate producer for next file
            producer = create_producer(args.broker)
    else:
        send_csv_to_kafka(args.csv, producer, args.topic, args.rate)


if __name__ == "__main__":
    main()
