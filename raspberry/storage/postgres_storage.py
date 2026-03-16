#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PostgreSQL Storage \u2014 Raspberry Pi Edge IDS.

Stores prediction results, confidence scores, and alert records in
PostgreSQL.  Tables are auto-created with appropriate indexes on first run.

Author  : Thai Nguyen Vu
Thesis  : Machine-Learning-Based Intrusion Detection on Edge Devices
"""

import os
import sys
import time
import json

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD


class PostgresStorage:
    """PostgreSQL adapter for storing IDS predictions and alerts."""

    def __init__(self):
        self.conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        self.conn.autocommit = True
        print(f"[OK] PostgreSQL connected: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")

    def init_tables(self):
        """Create tables if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp DOUBLE PRECISION NOT NULL,
                    prediction INTEGER NOT NULL,
                    confidence REAL,
                    label VARCHAR(20),
                    inference_time_ms REAL,
                    raw_features JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    alert_type VARCHAR(50) NOT NULL,
                    message TEXT,
                    confidence REAL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Create indexes for common queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                ON predictions (timestamp DESC);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_prediction
                ON predictions (prediction);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_created
                ON alerts (created_at DESC);
            """)
        print("[OK] PostgreSQL tables initialized")

    def store_prediction(self, timestamp, prediction, confidence,
                         label, inference_time_ms, raw_features=None):
        """Store a single prediction result."""
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO predictions
                   (timestamp, prediction, confidence, label, inference_time_ms, raw_features)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (timestamp, prediction, confidence, label,
                 inference_time_ms, json.dumps(raw_features) if raw_features else None),
            )

    def store_alert(self, alert_type, message, confidence=None):
        """Store an alert record."""
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO alerts (alert_type, message, confidence)
                   VALUES (%s, %s, %s)""",
                (alert_type, message, confidence),
            )

    def get_recent_predictions(self, limit=100):
        """Retrieve recent predictions for dashboard display."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()

    def get_attack_count(self, since_seconds=3600):
        """Count attacks in the last N seconds."""
        since = time.time() - since_seconds
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM predictions WHERE prediction = 1 AND timestamp >= %s",
                (since,),
            )
            return cur.fetchone()[0]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("[OK] PostgreSQL connection closed")
