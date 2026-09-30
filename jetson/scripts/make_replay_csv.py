#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the Kafka replay CSV from the leakage-free test split.

The sender replays sender/replay_cicids2017.csv row by row, and the edge tiers
read the columns in model/anomaly_feature_columns.json (gate) and
model/feature_columns.json (classifier), so the replay carries both. Rows are a
uniform random sample of the CICIDS2017 test split (no row seen in training),
so the class mix is the test split's own.
Labels go to a sidecar file with the same row order: the sender forwards every
column as a number, so a text label column cannot travel with the features.

    python scripts/make_replay_csv.py --rows 10000
"""

import argparse
import json
import os

import pyarrow.dataset as ds

JETSON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_ROOT = os.path.dirname(JETSON_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", default=os.path.join(THESIS_ROOT, "data", "test_data.parquet"))
    parser.add_argument("--features", default=os.path.join(JETSON_DIR, "model", "feature_columns.json"))
    parser.add_argument("--gate-features", default=os.path.join(JETSON_DIR, "model", "anomaly_feature_columns.json"))
    parser.add_argument("--out", default=os.path.join(JETSON_DIR, "sender", "replay_cicids2017.csv"))
    parser.add_argument("--labels-out", default=os.path.join(JETSON_DIR, "sender", "replay_cicids2017_labels.csv"))
    args = parser.parse_args()

    # The gate scores its own (larger) feature list and the classifier its SHAP
    # subset, so the replay carries the union: a column missing from a message
    # is silently read as 0 by the edge tiers.
    features = []
    for path in (args.gate_features, args.features):
        with open(path) as fh:
            features += [c for c in json.load(fh) if c not in features]

    table = ds.dataset(args.test, format="parquet").to_table(columns=features + ["label", "label_binary"])
    df = table.to_pandas().sample(n=args.rows, random_state=args.seed).reset_index(drop=True)

    df[features].to_csv(args.out, index=False)
    df[["label", "label_binary"]].to_csv(args.labels_out, index=False)

    attack_share = df["label_binary"].mean() * 100
    print(f"[OK] {args.rows:,} rows x {len(features)} features -> {args.out}")
    print(f"[OK] labels -> {args.labels_out} ({attack_share:.1f}% attack)")


if __name__ == "__main__":
    main()
