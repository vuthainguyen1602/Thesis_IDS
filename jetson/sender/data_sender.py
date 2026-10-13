#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replay a CICIDS2017 CSV into Kafka at a fixed rate.

Each message carries ``_timestamp`` (send time, for end-to-end latency) and
``_row`` (the CSV row index, which joins verdicts to the replay labels).

Pacing is deadline-based: row *i* is sent at ``start + i / rate``, so the
achieved rate matches the requested one up to several thousand rows/s. The
old loop slept ``1/rate`` after every row and topped out near 80 rows/s.
With ``--loop`` the file is replayed from its first row again once the end
is reached; ``--duration`` stops the sender after that many seconds and
prints a machine-readable ``SENDER_RESULT`` line with the rows sent.
"""

import os
import sys
import json
import time
import signal
import argparse
import csv

from kafka import KafkaProducer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, DATA_CSV_PATH, SEND_RATE


def create_producer(bootstrap_servers):
    # Rows are pre-serialised (see load_rows); the serializer is the identity.
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v,
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks=1,
        retries=3,
        batch_size=16384,
        linger_ms=10,
    )
    print(f"[OK] Kafka Producer connected to {bootstrap_servers}")
    return producer


def clean_column_name(name):
    result = name.strip().lower()
    for ch in [" ", ".", "-", "/", "(", ")"]:
        result = result.replace(ch, "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def load_rows(csv_path):
    """Parse the CSV once and return each row as a JSON object minus its
    closing brace, so the per-send work is appending two fields."""
    prefixes = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cleaned_fieldnames = {name: clean_column_name(name) for name in reader.fieldnames}
        for row in reader:
            cleaned_row = {}
            for original_name, value in row.items():
                clean_name = cleaned_fieldnames[original_name]
                try:
                    if "." in value:
                        cleaned_row[clean_name] = float(value)
                    else:
                        cleaned_row[clean_name] = int(value)
                except (ValueError, TypeError):
                    cleaned_row[clean_name] = value
            prefixes.append(json.dumps(cleaned_row)[:-1])
    return prefixes


class _Stop(Exception):
    pass


def send_csv_to_kafka(csv_path, producer, topic, rate=100, duration=0.0, loop=False):
    print(f"\n{'=' * 60}")
    print(f"  DATA SENDER")
    print(f"  CSV:   {csv_path}")
    print(f"  Topic: {topic}")
    print(f"  Rate:  {rate} rows/second" + (f" for {duration:g}s" if duration > 0 else "")
          + (" (looping)" if loop else ""))
    print(f"{'=' * 60}\n")

    rows = load_rows(csv_path)
    n_rows = len(rows)
    interval = 1.0 / rate if rate > 0 else 0.0

    def _on_term(signum, frame):
        raise _Stop()

    signal.signal(signal.SIGTERM, _on_term)

    total_sent = 0
    start_time = time.time()
    deadline = start_time + duration if duration > 0 else None
    stopped_by = "end_of_csv"

    try:
        while True:
            for row_idx in range(n_rows):
                now = time.time()
                if deadline is not None and now >= deadline:
                    stopped_by = "duration"
                    raise _Stop()
                if interval > 0:
                    target = start_time + total_sent * interval
                    if target > now:
                        time.sleep(target - now)
                        now = time.time()
                value = (rows[row_idx] + f',"_timestamp":{now!r},"_row":{row_idx}}}').encode("utf-8")
                producer.send(topic, key=str(total_sent), value=value)
                total_sent += 1

                if total_sent % 1000 == 0:
                    elapsed = time.time() - start_time
                    actual_rate = total_sent / elapsed if elapsed > 0 else 0
                    print(f"  Sent {total_sent:,} rows | "
                          f"Rate: {actual_rate:.1f} rows/s | "
                          f"Elapsed: {elapsed:.1f}s")
            if not loop:
                break

    except KeyboardInterrupt:
        stopped_by = "interrupt"
        print(f"\n  [WARN] Interrupted by user after {total_sent:,} rows")
    except _Stop:
        if stopped_by != "duration":
            stopped_by = "sigterm"
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start_time
        avg_rate = total_sent / elapsed if elapsed > 0 else 0.0
        print(f"\n{'=' * 60}")
        print(f"  SENDER COMPLETE ({stopped_by})")
        print(f"  Total sent:  {total_sent:,} rows")
        print(f"  Total time:  {elapsed:.1f}s")
        print(f"  Avg rate:    {avg_rate:.1f} rows/s")
        print(f"{'=' * 60}")
        print("SENDER_RESULT " + json.dumps({
            "rows_sent": total_sent, "elapsed_s": round(elapsed, 3),
            "rate_rps": round(avg_rate, 2), "csv_rows": n_rows,
            "requested_rate_rps": rate, "stopped_by": stopped_by,
        }), flush=True)


def main():
    parser = argparse.ArgumentParser(description="IDS Data Sender - CSV to Kafka")
    parser.add_argument("--csv", type=str, default=DATA_CSV_PATH,
                        help="Path to CSV file or directory")
    parser.add_argument("--rate", type=float, default=SEND_RATE,
                        help="Send rate (rows per second); <= 0 sends as fast as possible")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Stop after this many seconds (0 = send the whole file)")
    parser.add_argument("--loop", action="store_true",
                        help="Replay the file from the start again until --duration elapses")
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
            producer = create_producer(args.broker)
    else:
        send_csv_to_kafka(args.csv, producer, args.topic, args.rate,
                          duration=args.duration, loop=args.loop)


if __name__ == "__main__":
    main()
