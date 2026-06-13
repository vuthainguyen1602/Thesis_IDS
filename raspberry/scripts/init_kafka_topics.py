#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_SUSPICIOUS_TOPIC


def create_topics(partitions: int = 2, replication_factor: int = 1):
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    topics = [
        NewTopic(name=KAFKA_TOPIC, num_partitions=partitions, replication_factor=replication_factor),
        NewTopic(name=KAFKA_SUSPICIOUS_TOPIC, num_partitions=partitions, replication_factor=replication_factor),
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Kafka topics for distributed IDS")
    parser.add_argument("--partitions", type=int, default=2, help="Number of partitions (>= number of Jetsons)")
    args = parser.parse_args()
    create_topics(partitions=max(2, args.partitions))
