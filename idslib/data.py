"""IDS data preparation: column cleaning, leakage-aware loading, stratified sampling."""
import os, re
from .core import *


def _clean_name(col_name: str) -> str:
    new_name = col_name.strip().lower()
    new_name = re.sub(r"[\s.\-/]+", "_", new_name)
    new_name = re.sub(r"[()]", "", new_name)
    new_name = re.sub(r"_+", "_", new_name)
    return new_name.strip("_")


# CSE-CIC-IDS2018 ships CICFlowMeter-v3 ABBREVIATED headers ("Tot Fwd Pkts",
# "TotLen Fwd Pkts", "Init Fwd Win Byts", ...) while CICIDS2017 uses the long
# names ("Total Fwd Packets", ...). Without harmonisation the cross-dataset
# feature intersection (ml_11) is computed on cleaned NAMES and would be nearly
# empty / semantically wrong. This map renames the cleaned 2018-style names to
# the 2017 canonical names; 2017 headers never use the abbreviated forms, so
# applying it unconditionally is a no-op for 2017 data.
CICIDS2018_TO_2017_ALIASES = {
    "dst_port": "destination_port",
    "src_port": "source_port",
    "tot_fwd_pkts": "total_fwd_packets",
    "tot_bwd_pkts": "total_backward_packets",
    "totlen_fwd_pkts": "total_length_of_fwd_packets",
    "totlen_bwd_pkts": "total_length_of_bwd_packets",
    "fwd_pkt_len_max": "fwd_packet_length_max",
    "fwd_pkt_len_min": "fwd_packet_length_min",
    "fwd_pkt_len_mean": "fwd_packet_length_mean",
    "fwd_pkt_len_std": "fwd_packet_length_std",
    "bwd_pkt_len_max": "bwd_packet_length_max",
    "bwd_pkt_len_min": "bwd_packet_length_min",
    "bwd_pkt_len_mean": "bwd_packet_length_mean",
    "bwd_pkt_len_std": "bwd_packet_length_std",
    "flow_byts_s": "flow_bytes_s",
    "flow_pkts_s": "flow_packets_s",
    "fwd_iat_tot": "fwd_iat_total",
    "bwd_iat_tot": "bwd_iat_total",
    "fwd_header_len": "fwd_header_length",
    "bwd_header_len": "bwd_header_length",
    "fwd_pkts_s": "fwd_packets_s",
    "bwd_pkts_s": "bwd_packets_s",
    "pkt_len_min": "min_packet_length",
    "pkt_len_max": "max_packet_length",
    "pkt_len_mean": "packet_length_mean",
    "pkt_len_std": "packet_length_std",
    "pkt_len_var": "packet_length_variance",
    "fin_flag_cnt": "fin_flag_count",
    "syn_flag_cnt": "syn_flag_count",
    "rst_flag_cnt": "rst_flag_count",
    "psh_flag_cnt": "psh_flag_count",
    "ack_flag_cnt": "ack_flag_count",
    "urg_flag_cnt": "urg_flag_count",
    "cwe_flag_cnt": "cwe_flag_count",
    "ece_flag_cnt": "ece_flag_count",
    "pkt_size_avg": "average_packet_size",
    "fwd_seg_size_avg": "avg_fwd_segment_size",
    "bwd_seg_size_avg": "avg_bwd_segment_size",
    "fwd_byts_b_avg": "fwd_avg_bytes_bulk",
    "fwd_pkts_b_avg": "fwd_avg_packets_bulk",
    "fwd_blk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_byts_b_avg": "bwd_avg_bytes_bulk",
    "bwd_pkts_b_avg": "bwd_avg_packets_bulk",
    "bwd_blk_rate_avg": "bwd_avg_bulk_rate",
    "subflow_fwd_pkts": "subflow_fwd_packets",
    "subflow_fwd_byts": "subflow_fwd_bytes",
    "subflow_bwd_pkts": "subflow_bwd_packets",
    "subflow_bwd_byts": "subflow_bwd_bytes",
    "init_fwd_win_byts": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
}


def _canonical_name(col_name: str) -> str:
    cleaned = _clean_name(col_name)
    return CICIDS2018_TO_2017_ALIASES.get(cleaned, cleaned)


def clean_column_names(df):
    # Single projection instead of one withColumnRenamed per column (avoids a
    # deep, O(n) logical plan for ~78 columns). Names are cleaned AND harmonised
    # to the CICIDS2017 canonical vocabulary (see CICIDS2018_TO_2017_ALIASES).
    aliased = [c for c in df.columns
               if _clean_name(c) in CICIDS2018_TO_2017_ALIASES]
    if aliased:
        print(f"  [INFO] Harmonised {len(aliased)} CICFlowMeter-v3 (IDS2018-style) "
              f"column names to CICIDS2017 canonical names.")
    return df.select(
        [F.col("`" + c + "`").alias(_canonical_name(c)) for c in df.columns]
    )


def handle_infinity_values(df):
    # Compute dtypes once (df.dtypes rebuilds the whole list on every call) and
    # apply all column rewrites in a single select rather than chaining one
    # withColumn per column.
    dtypes = dict(df.dtypes)
    select_exprs = []
    for col_name in df.columns:
        if dtypes[col_name] in ["double", "float"]:
            c = F.col(col_name)
            select_exprs.append(
                F.when(
                    c.isNull()
                    | F.isnan(c)
                    | (c == float("inf"))
                    | (c == float("-inf")),
                    None,
                ).otherwise(c).alias(col_name)
            )
        else:
            select_exprs.append(F.col(col_name))
    return df.select(select_exprs)


def align_schema(df, ref_columns: list):
    for c in ref_columns:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(StringType()))
    return df.select(ref_columns)



def _leaky_port_cols() -> list:
    """Port columns excluded from the feature set to prevent label leakage.

    In CICIDS2017 each attack class was generated against fixed target
    ports/services, so ``destination_port`` (and ``source_port`` when present)
    acts as a near-label shortcut: a model can memorise "port X -> attack",
    inflating F1 toward 1.0 while failing to generalise — which also corrupts
    the robustness and concept-drift evaluation.

    Excluded by default. Set ``IDS_KEEP_PORT_FEATURES=1`` to keep them for an
    explicit with-port vs. without-port ablation.
    """
    if os.environ.get("IDS_KEEP_PORT_FEATURES", "0") == "1":
        return []
    return ["destination_port", "source_port", "src_port", "dst_port"]


def load_and_prepare_data(
    spark, data_dir=None
) -> tuple:
    output_dir = data_dir or resolve_data_dir()
    train_path = os.path.join(output_dir, "train_data.parquet")
    test_path = os.path.join(output_dir, "test_data.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Parquet data not found. Run ml_00_prepare_cicids2017.py first!\n"
            f"  Expected: {train_path}\n"
            f"  Expected: {test_path}"
        )

    print("=" * 60)
    print("  LOADING DATA FROM PARQUET")
    print("=" * 60)

    train_df = spark.read.parquet(train_path)
    test_df = spark.read.parquet(test_path)

    train_df = train_df.cache()
    test_df = test_df.cache()

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"  Training set: {train_count:,} samples")
    print(f"  Test set:     {test_count:,} samples")

    # Lazy union (no eager cache/materialize): only ml_07 consumes it, and it
    # caches locally where the drift simulation needs it. Avoids holding a 3rd
    # full copy of the data in memory for the other scripts.
    df = train_df.unionByName(test_df)

    # IDS_KEEP_PORT_FEATURES=1 deliberately RE-INCLUDES the leaky port features
    # (destination_port, ...). This is ONLY for the leakage ablation
    # (ml_10_leakage_ablation.py); every reported pipeline keeps it at 0.
    keep_ports = os.environ.get("IDS_KEEP_PORT_FEATURES", "0") == "1"
    exclude_cols = ["label", "label_binary", "source_ip", "destination_ip",
                    "flow_id", "timestamp", "protocol"]
    if not keep_ports:
        exclude_cols += _leaky_port_cols()
    # Derive feature list from the train schema (dtypes computed once instead of
    # rebuilt per column); train and union share the same schema.
    dtypes = dict(train_df.dtypes)
    feature_cols = [
        c for c in train_df.columns
        if c not in exclude_cols
        and dtypes[c] in ["double", "float", "int", "bigint"]
    ]
    if keep_ports:
        print("  [ABLATION] Port features KEPT (IDS_KEEP_PORT_FEATURES=1) — "
              "LEAKY configuration, for ablation only.")
    else:
        print("  [INFO] Port features excluded to avoid label leakage "
              "(set IDS_KEEP_PORT_FEATURES=1 for ablation).")
    print(f"  Numeric features: {len(feature_cols)}")
    print("=" * 60)

    return df, train_df, test_df, feature_cols



def stratified_sample(df, select_cols: list, label_col: str = "label_binary",
                      sample_size: int = 2000, seed: int = 42):
    """Return a random, class-stratified sample of about ``sample_size`` rows.

    Replaces ``df.select(cols).limit(n)``, which returns an arbitrary slice
    determined by partition order — often class-skewed (e.g. all-benign) and
    therefore an unrepresentative basis for SHAP global importance. The sampling
    budget is split evenly across label strata and turned into a per-class
    fraction, so every class (including rare attack types) is represented; the
    draw is random across the whole DataFrame. No trailing ``limit`` is applied,
    avoiding the partition-order bias that would re-skew the final class balance.
    """
    selected = df.select(select_cols)
    total = selected.count()
    if total <= sample_size:
        return selected
    counts = {r[label_col]: r["count"]
              for r in df.groupBy(label_col).count().collect()}
    per_class_budget = max(1, sample_size // max(1, len(counts)))
    fractions = {lbl: min(1.0, per_class_budget / c)
                 for lbl, c in counts.items() if c > 0}
    return selected.sampleBy(label_col, fractions=fractions, seed=seed)


