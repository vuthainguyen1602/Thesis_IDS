#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
DATA PREPARATION - RoEduNet-SIMARGL2021 Dataset
================================================================================
Run this script ONCE to:
  1. Load RoEduNet-SIMARGL2021 CSV file(s)
  2. Clean column names, handle infinity/NaN values
  3. Sample data to manageable size (~2-3M rows if dataset is large)
  4. Create binary labels (Normal=0, Attack=1)
  5. Split into train/test sets (80/20)
  6. Save as parquet files for fast loading

Dataset: https://www.kaggle.com/datasets/simargl/roedunet-simargl2021

Output:
  data/train_data.parquet
  data/test_data.parquet
================================================================================
"""

import os
from shared_utils import (
    create_spark_session,
    clean_column_names,
    handle_infinity_values,
    F, col, when, StringType,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_PATH = "/Users/thainguyenvu/Desktop/roedunet-simargl2021"
OUTPUT_DIR = "/Users/thainguyenvu/Desktop/Thesis_IDS/Thesis_IDS_RoEduNet/data"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train_data.parquet")
TEST_PATH = os.path.join(OUTPUT_DIR, "test_data.parquet")

# Maximum number of rows to use (set to None to use all data)
# RoEduNet has ~45M rows; we sample to ~2.5M for local training feasibility
MAX_ROWS = 2_500_000

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":

    spark = create_spark_session("IDS_DataPrep_RoEduNet")

    print("=" * 60)
    print("LOADING RoEduNet-SIMARGL2021 DATA")
    print("=" * 60)

    # Load all CSV files from the input directory
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("escape", '"')
        .option("multiLine", True)
        .csv(INPUT_PATH)
    )

    row_count = df.count()
    print(f"  Total rows loaded: {row_count:,}")
    print(f"  Columns ({len(df.columns)}): {df.columns}")

    # --- Clean column names ---
    df = clean_column_names(df)
    print(f"  Columns after cleaning: {df.columns}")

    # --- Handle infinity/NaN ---
    df = handle_infinity_values(df)

    # --- Remove duplicates and nulls ---
    print(f"\n  Rows before cleaning: {row_count:,}")
    df = df.dropDuplicates()
    df = df.dropna()
    cleaned_count = df.count()
    print(f"  Rows after cleaning:  {cleaned_count:,}")

    # --- Sample if dataset is too large ---
    if MAX_ROWS and cleaned_count > MAX_ROWS:
        sample_fraction = MAX_ROWS / cleaned_count
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=42)
        sampled_count = df.count()
        print(f"  Sampled to ~{MAX_ROWS:,} rows: actual = {sampled_count:,}")
    else:
        sampled_count = cleaned_count

    # --- Detect label column ---
    # RoEduNet uses a 'label' column (or similar) with values like:
    # 'Normal flow', 'SYN Scan', 'Denial of Service SlowLoris', 'Denial of Service R-U-Dead-Yet'
    # We need to identify it dynamically
    label_col_name = None
    for candidate in ["label", "attack_type", "class", "category"]:
        if candidate in df.columns:
            label_col_name = candidate
            break

    if label_col_name is None:
        # Try to find a string column that looks like a label
        string_cols = [c for c in df.columns if dict(df.dtypes)[c] == "string"]
        if string_cols:
            label_col_name = string_cols[-1]  # Usually the last string column
            print(f"  Auto-detected label column: '{label_col_name}'")
        else:
            raise ValueError("Could not find a label column in the dataset!")

    print(f"\n  Label column: '{label_col_name}'")
    print("  Label distribution:")
    df.groupBy(label_col_name).count().orderBy(F.desc("count")).show(20, truncate=False)

    # --- Rename label column and create binary label ---
    if label_col_name != "label":
        df = df.withColumnRenamed(label_col_name, "label")

    # Binary: Normal/Benign = 0, everything else (attacks) = 1
    df = df.withColumn(
        "label_binary",
        when(
            (col("label").contains("Normal")) | (col("label").contains("BENIGN")) | (col("label").contains("normal")),
            0,
        ).otherwise(1),
    )

    print("  Binary label distribution:")
    df.groupBy("label_binary").count().orderBy("label_binary").show()

    # --- Identify numeric feature columns ---
    exclude_cols = ["label", "label_binary", "src_ip", "dst_ip",
                    "time_first", "time_last", "proto", "tag"]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and dict(df.dtypes)[c] in ["double", "float", "int", "bigint", "long"]
    ]
    print(f"  Number of numeric features: {len(feature_cols)}")
    print(f"  Features: {feature_cols}")

    # --- Split train/test ---
    df = df.cache()
    df.count()

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"\n  Training set: {train_count:,} samples")
    print(f"  Test set:     {test_count:,} samples")

    # --- Save to parquet ---
    print(f"\n  Saving to parquet...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df.write.mode("overwrite").parquet(TRAIN_PATH)
    print(f"  Saved: {TRAIN_PATH}")

    test_df.write.mode("overwrite").parquet(TEST_PATH)
    print(f"  Saved: {TEST_PATH}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED (RoEduNet-SIMARGL2021)")
    print("=" * 60)
    print("You can now run experiment files (exp0, exp1, exp2).")

    spark.stop()
    print("Spark Session closed.")
