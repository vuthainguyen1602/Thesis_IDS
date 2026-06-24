#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from shared_utils import (
    create_spark_session,
    clean_column_names,
    handle_infinity_values,
    align_schema,
    _leaky_port_cols,
    F, col, when,
)

import glob as _glob

# Dataset selection. Default = CICIDS2017. A SECOND dataset (e.g.
# CSE-CIC-IDS2018, CIC-IoT2023 — also CICFlowMeter CSVs) can be prepared
# WITHOUT editing this file: set IDS_DATASET=<name> for the label/output dir and
# IDS_CSV_GLOB to a glob that matches its CSV files. The cleaning, leakage-aware
# dedup and split logic below are dataset-agnostic (CICFlowMeter schema).
DATASET = os.environ.get("IDS_DATASET", "cicids2017")
INPUT_PATH = os.environ.get("IDS_RAW_DATA_DIR", os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "ids-2017"))
OUTPUT_DIR = os.environ.get("IDS_DATA_DIR", os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "data"))
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train_data.parquet")
TEST_PATH = os.path.join(OUTPUT_DIR, "test_data.parquet")
PARQUET_PARTITIONS = int(os.environ.get("IDS_PARQUET_PARTITIONS", "8"))

_DEFAULT_CICIDS2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

_csv_glob = os.environ.get("IDS_CSV_GLOB")
if _csv_glob:
    CSV_FILES = [os.path.basename(p) for p in sorted(_glob.glob(os.path.join(INPUT_PATH, _csv_glob)))]
    if not CSV_FILES:
        raise FileNotFoundError(f"IDS_CSV_GLOB='{_csv_glob}' matched no files in {INPUT_PATH}")
else:
    CSV_FILES = _DEFAULT_CICIDS2017_FILES
print(f"[INFO] Dataset='{DATASET}' | {len(CSV_FILES)} CSV files from {INPUT_PATH}")


if __name__ == "__main__":
    # Raw CSV merge runs on Mac only (before sync to Jetsons)
    os.environ["IDS_ALLOW_LOCAL_SPARK"] = "1"

    spark = create_spark_session("IDS_Data_Preparation")

    print("=" * 60)
    print("START MERGING CICIDS2017 DATA")
    print("=" * 60)

    merged_df = None
    unified_columns = None

    for i, filename in enumerate(CSV_FILES):
        file_path = os.path.join(INPUT_PATH, filename)
        print(f"\n[{i+1}/{len(CSV_FILES)}] Processing: {filename}")

        try:
            df = (
                spark.read.option("header", True)
                .option("inferSchema", True)
                .option("escape", '"')
                .csv(file_path)
            )

            row_count = df.count()
            print(f"  Read successfully: {row_count:,} rows, {len(df.columns)} columns")

            df = clean_column_names(df)
            df = handle_infinity_values(df)

            if merged_df is None:
                unified_columns = df.columns
                merged_df = df
                print(f"  Initialized schema with {len(unified_columns)} columns")
                continue

            df = align_schema(df, unified_columns)
            merged_df = merged_df.unionByName(df)
            print(f"  Merged into total DataFrame")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue

    if merged_df is None:
        raise RuntimeError(
            "No CSV files could be read from the input directory. "
            "Check the dataset path / IDS_CSV_GLOB pattern."
        )

    print("\n" + "=" * 60)
    print("MERGE COMPLETED")
    print("=" * 60)

    print(f"\nTotal rows before cleaning: {merged_df.count():,}")
    merged_df = merged_df.dropDuplicates()
    merged_df = merged_df.dropna()
    print(f"Total rows after exact dedup + dropna: {merged_df.count():,}")

    df = merged_df.withColumn(
        "label_binary",
        when(col("label") == "BENIGN", 0).otherwise(1),
    )

    print("\nLabel distribution:")
    df.groupBy("label").count().orderBy(F.desc("count")).show(20, truncate=False)

    print("Binary label distribution:")
    df.groupBy("label_binary").count().orderBy("label_binary").show()

    exclude_cols = ["label", "label_binary", "source_ip", "destination_ip",
                    "flow_id", "timestamp", "protocol"] + _leaky_port_cols()
    dtypes = dict(df.dtypes)
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and dtypes[c] in ["double", "float", "int", "bigint"]
    ]
    print(f"Number of numeric features: {len(feature_cols)}")

    # ── Near-duplicate removal on FEATURE columns ────────────────────────────
    # CICIDS2017 contains many flows that are identical on all feature columns
    # but were not removed by exact dropDuplicates() (e.g. differing
    # identifier/timestamp fields). A random train/test split then places such
    # near-duplicates on both sides, leaking and inflating test metrics. Dedup
    # on feature columns BEFORE splitting closes this. Disable with
    # IDS_DEDUP_ON_FEATURES=0 to reproduce the legacy (leakier) behaviour.
    if os.environ.get("IDS_DEDUP_ON_FEATURES", "1") == "1":
        before = df.count()
        df = df.dropDuplicates(feature_cols)
        after = df.count()
        print(f"Feature-level dedup: {before:,} -> {after:,} "
              f"(removed {before - after:,} near-duplicate flows)")

    # ── Optional stratified subsample (for very large datasets, e.g. CSE-CIC-
    # IDS2018 ~16M flows on 8GB Jetsons). IDS_SAMPLE_FRAC in (0,1) keeps a
    # class-proportional random subset so the distribution — including rare
    # attack types — is preserved while the data fits the edge hardware.
    sample_frac = float(os.environ.get("IDS_SAMPLE_FRAC", "1.0"))
    if 0.0 < sample_frac < 1.0:
        before = df.count()
        label_vals = [r["label"] for r in df.select("label").distinct().collect()]
        fractions = {lbl: sample_frac for lbl in label_vals}
        df = df.sampleBy("label", fractions=fractions, seed=42)
        print(f"Stratified subsample (IDS_SAMPLE_FRAC={sample_frac}): "
              f"{before:,} -> {df.count():,} rows (class proportions preserved).")

    df = df.cache()
    df.count()

    # ── Train/test split: random (default) or temporal ───────────────────────
    # IDS_SPLIT_MODE=temporal orders by timestamp and uses the earliest 80% for
    # training and the latest 20% for testing — a more realistic IDS evaluation
    # (train on past, detect future) and a prerequisite for honest drift claims.
    split_mode = os.environ.get("IDS_SPLIT_MODE", "random").strip().lower()
    if split_mode == "temporal" and "timestamp" in df.columns:
        ts_df = (
            df.withColumn("_ts_num", F.unix_timestamp("timestamp").cast("double"))
            .filter(F.col("_ts_num").isNotNull())
        )
        n_ts = ts_df.count()
        if n_ts > 0:
            n_dropped = df.count() - n_ts
            if n_dropped > 0:
                print(f"[WARN] Temporal split drops {n_dropped:,} rows with "
                      f"missing/unparseable timestamps (kept {n_ts:,}). Use the "
                      f"default random split to retain all rows.")
            cutoff = ts_df.approxQuantile("_ts_num", [0.8], 0.005)[0]
            train_df = ts_df.filter(F.col("_ts_num") <= cutoff).drop("_ts_num")
            test_df = ts_df.filter(F.col("_ts_num") > cutoff).drop("_ts_num")
            print(f"[INFO] Temporal split at timestamp quantile 0.8 (cutoff={cutoff}).")
        else:
            print("[WARN] IDS_SPLIT_MODE=temporal but no valid timestamps; using random split.")
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    else:
        if split_mode == "temporal":
            print("[WARN] IDS_SPLIT_MODE=temporal but no 'timestamp' column; using random split.")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"\nTraining set: {train_count:,} samples")
    print(f"Test set:     {test_count:,} samples")

    print(f"\nSaving to parquet...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in (TRAIN_PATH, TEST_PATH):
        if os.path.exists(path):
            shutil.rmtree(path)

    train_df.coalesce(PARQUET_PARTITIONS).write.mode("overwrite").parquet(TRAIN_PATH)
    print(f"  Saved: {TRAIN_PATH}")

    test_df.coalesce(PARQUET_PARTITIONS).write.mode("overwrite").parquet(TEST_PATH)
    print(f"  Saved: {TEST_PATH}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print("You can now run any ML experiment script (ml_01 … ml_08).")

    spark.stop()
    print("Spark Session closed.")
