#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from shared_utils import (
    create_spark_session,
    clean_column_names,
    handle_infinity_values,
    align_schema,
    F, col, when, StringType,
)

INPUT_PATH = os.environ.get("IDS_RAW_DATA_DIR", os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "ids-2017"))
OUTPUT_DIR = os.environ.get("IDS_DATA_DIR", os.path.join(os.environ.get("IDS_ROOT", os.path.dirname(os.path.abspath(__file__))), "data"))
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train_data.parquet")
TEST_PATH = os.path.join(OUTPUT_DIR, "test_data.parquet")

CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


if __name__ == "__main__":

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
                .option("multiLine", True)
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

    print("\n" + "=" * 60)
    print("MERGE COMPLETED")
    print("=" * 60)

    print(f"\nTotal rows before cleaning: {merged_df.count():,}")
    merged_df = merged_df.dropDuplicates()
    merged_df = merged_df.dropna()
    print(f"Total rows after cleaning:  {merged_df.count():,}")

    df = merged_df.withColumn(
        "label_binary",
        when(col("label") == "BENIGN", 0).otherwise(1),
    )

    print("\nLabel distribution:")
    df.groupBy("label").count().orderBy(F.desc("count")).show(20, truncate=False)

    print("Binary label distribution:")
    df.groupBy("label_binary").count().orderBy("label_binary").show()

    exclude_cols = ["label", "label_binary", "source_ip", "destination_ip",
                    "flow_id", "timestamp", "protocol"]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and dict(df.dtypes)[c] in ["double", "float", "int", "bigint"]
    ]
    print(f"Number of numeric features: {len(feature_cols)}")

    df = df.cache()
    df.count()

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"\nTraining set: {train_count:,} samples")
    print(f"Test set:     {test_count:,} samples")

    print(f"\nSaving to parquet...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df.write.mode("overwrite").parquet(TRAIN_PATH)
    print(f"  Saved: {TRAIN_PATH}")

    test_df.write.mode("overwrite").parquet(TEST_PATH)
    print(f"  Saved: {TEST_PATH}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print("You can now run any experiment file (exp0, exp1, exp2, exp3, exp4).")

    spark.stop()
    print("Spark Session closed.")
