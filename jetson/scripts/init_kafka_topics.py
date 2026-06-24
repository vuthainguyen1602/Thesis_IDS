#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import NoBrokersAvailable, TopicAlreadyExistsError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_SUSPICIOUS_TOPIC


def create_topics(
    partitions: int = 2,
    replication_factor: int = 1,
    bootstrap_servers: str | None = None,
    retries: int = 30,
    retry_delay: float = 2.0,
):
    servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=servers,
                request_timeout_ms=10000,
            )
            topics = [
                NewTopic(
                    name=KAFKA_TOPIC,
                    num_partitions=partitions,
                    replication_factor=replication_factor,
                ),
                NewTopic(
                    name=KAFKA_SUSPICIOUS_TOPIC,
                    num_partitions=partitions,
                    replication_factor=replication_factor,
                ),
            ]
            try:
                admin.create_topics(new_topics=topics, validate_only=False)
                print(f"[OK] Created topics with {partitions} partitions:")
                print(f"  - {KAFKA_TOPIC}")
                print(f"  - {KAFKA_SUSPICIOUS_TOPIC}")
            except TopicAlreadyExistsError:
                print("[INFO] Topics already exist")
            finally:
                admin.close()
            return
        except NoBrokersAvailable as exc:
            last_error = exc
            if attempt < retries:
                print(
                    f"[WAIT] Kafka not ready at {servers} "
                    f"({attempt}/{retries}) — retry in {retry_delay:.0f}s..."
                )
                time.sleep(retry_delay)
            else:
                break

    print(f"[ERR] No Kafka broker at {servers}")
    print("      1. docker compose ps   (kafka must be Up)")
    print("      2. docker compose logs kafka | tail -30")
    print("      3. On Mac try: python scripts/init_kafka_topics.py --bootstrap localhost:9092")
    raise last_error or NoBrokersAvailable()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Kafka topics for distributed IDS")
    parser.add_argument("--partitions", type=int, default=2, help="Number of partitions (>= number of Jetsons)")
    parser.add_argument(
        "--bootstrap",
        dest="bootstrap_servers",
        default=None,
        help="Override bootstrap servers (Mac: localhost:9092)",
    )
    parser.add_argument("--retries", type=int, default=30, help="Retry count while Kafka starts")
    args = parser.parse_args()
    create_topics(
        partitions=max(2, args.partitions),
        bootstrap_servers=args.bootstrap_servers,
        retries=args.retries,
    )
