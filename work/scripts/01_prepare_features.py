from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_utils import (
    PROCESSED_DIR,
    RAW_PATH,
    ensure_dirs,
    write_json,
)


# ── Config ───────────────────────────────────────────────────────────────────

POS_BUCKETS = [0, 3, 5, 10, 20, 50, 100, 9999]
POS_LABELS = ["1-3", "4-5", "6-10", "11-20", "21-50", "51-100", "100+"]


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FlyRank refresh feature vector.")
    parser.add_argument(
        "--input",
        default=str(RAW_PATH.with_name("content_refresh_3m.parquet")),
        help="Raw Parquet export.",
    )
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DIR / "refresh_feature_vector.csv"),
        help="Prepared feature-vector CSV.",
    )
    return parser.parse_args()


# ── Feature Engineering ──────────────────────────────────────────────────────

def add_has_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["has_word_count"] = (df["word_count"] > 0).astype(int)
    df["has_search_volume"] = (df["search_volume"] > 0).astype(int)
    df["has_backlinks"] = (df["backlinks"] > 0).astype(int)
    df["has_category_count"] = (df["category_count"] > 0).astype(int)
    df["has_competition"] = (df["competition"] > 0).astype(int)
    df["has_cpc"] = (df["cpc"] > 0).astype(int)
    return df


def add_content_features(df: pd.DataFrame) -> pd.DataFrame:
    df["content_depth"] = np.log1p(df["word_count"].fillna(0)) * df["has_word_count"]
    df["market_difficulty"] = df["has_search_volume"] * df["competition"].fillna(0)
    # df["impressions_per_day"] = df["total_impressions"] / df["days_with_data"].clip(lower=1)
    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df["wasted_age"] = df["content_age_days"] * (1 - df["ever_optimized"])
    df["keyword_opportunity"] = (
        df["search_volume"] / df["competition"].replace(0, 0.01)
    ).fillna(0)
    df["commercial_value"] = df["cpc"] * df["has_cpc"]
    df["depth_vs_demand"] = (
        np.log1p(df["word_count"]) / np.log1p(df["search_volume"].fillna(1))
    )
    return df


def add_freshness_features(df: pd.DataFrame) -> pd.DataFrame:
    df["freshness_score"] = 1 / (1 + df["days_since_update"].fillna(365))
    df["optimization_recency"] = 1 / (1 + df["days_since_optimized"].fillna(999))
    df["content_maturity"] = (
        df["content_age_days"] / (1 + df["days_since_update"].fillna(365))
    ).clip(upper=10)
    df["optimized_and_fresh"] = df["ever_optimized"] * df["freshness_score"]
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # df["backlink_efficiency"] = df["backlinks"] / (df["total_impressions"] / 1000 + 0.1)
    df["breadth_vs_depth"] = df["category_count"].fillna(0) / np.log1p(df["word_count"].fillna(0) + 1)
    df["structured_content"] = (
        df["has_keyword_data"] * (df["word_count"].fillna(0) > 500).astype(int)
    )
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df["age_squared"] = df["content_age_days"] ** 2 / 1000
    df["log_age"] = np.log1p(df["content_age_days"])
    df["is_old"] = (df["content_age_days"] > 365).astype(int)
    df["is_new"] = (df["content_age_days"] < 90).astype(int)
    return df


def add_visibility_features(df: pd.DataFrame) -> pd.DataFrame:
    # df["lifetime_impressions_est"] = df["impressions_per_day"] * df["content_age_days"]
    # df["is_high_visibility"] = (df["total_impressions"] > 1000).astype(int)

    has_cols = [c for c in df.columns if c.startswith("has_")]
    df["data_completeness"] = df[has_cols].sum(axis=1) / len(has_cols)
    return df


def add_position_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df["pos_bucket"] = pd.cut(
        df["avg_position"],
        bins=POS_BUCKETS,
        labels=POS_LABELS,
    )
    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    # Category-relative features
    cat_word_median = df.groupby("content_type")["word_count"].transform("median")
    df["cat_words_ratio"] = df["word_count"] / cat_word_median.replace(0, 1)

    intent_backlink_median = df.groupby("main_intent")["backlinks"].transform("median")
    df["intent_backlinks_ratio"] = df["backlinks"] / intent_backlink_median.replace(0, 1)

    category_word_mean = df.groupby("content_type")["word_count"].transform("mean")
    df["diff_from_cat_avg_words"] = df["word_count"] - category_word_mean

    intent_backlink_mean = df.groupby("main_intent")["backlinks"].transform("mean")
    df["diff_from_intent_avg_backlinks"] = df["backlinks"] - intent_backlink_mean

    # Age / optimization ratios
    df["unoptimized_ratio"] = df["days_since_optimized"] / (df["content_age_days"] + 1)
    df["backlinks_per_age"] = df["backlinks"] / (df["content_age_days"] + 1)
    df["update_optimization_gap"] = df["days_since_update"] - df["days_since_optimized"]
    df["words_per_age_day"] = df["word_count"] / (df["content_age_days"] + 1)
    # df["market_capture_ratio"] = df["total_impressions"] / (df["search_volume"] + 1)
    df["commercial_potential"] = (
        np.log1p(df["cpc"] * df["search_volume"]) * df["keyword_opportunity"]
    )

    # Bucket-relative features: how does this page compare to peers in the
    # same position bucket?  This gives context without leaking the bucket
    # identity as a direct categorical feature.
    bucket_cols = [
        "word_count", "backlinks", "content_age_days",
        # "days_since_update", 
        "search_volume", "cpc",
    ]
    for col in bucket_cols:
        bucket_median = df.groupby("pos_bucket")[col].transform("median")
        df[f"{col}_vs_bucket_median"] = df[col] / bucket_median.replace(0, 1)

        bucket_mean = df.groupby("pos_bucket")[col].transform("mean")
        df[f"{col}_vs_bucket_mean"] = df[col] / bucket_mean.replace(0, 1)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-feature interactions that capture compound effects."""
    df["freshness_x_demand"] = df["freshness_score"] * df["search_volume"]
    df["depth_vs_competition"] = df["content_depth"] / (df["competition"] + 0.01)
    df["age_x_optimization"] = df["content_age_days"] * df["ever_optimized"]
    df["maturity_x_freshness"] = df["content_maturity"] * df["freshness_score"]
    df["opportunity_x_value"] = df["keyword_opportunity"] * df["commercial_value"]
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ensure_dirs()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw input not found: {input_path}")

    # Load
    df = pd.read_parquet(input_path)
    initial_rows = len(df)
    print(f"Loaded {initial_rows:,} rows from {input_path}")

    # Filter
    df = df[df["total_impressions"] >= 100].copy().reset_index(drop=True)
    print(f"After visibility filter (impressions >= 100): {len(df):,} rows")

    df = df.drop_duplicates(subset=["content_hash_id"]).reset_index(drop=True)
    print(f"After dedup: {len(df):,} rows")

    df["days_since_update"] = df["days_since_update"].clip(lower=0)
    df["days_since_optimized"] = df["days_since_optimized"].clip(lower=0)

    # Clean numeric
    numeric_cols = [
        "avg_position", "total_clicks", "total_impressions", "ctr",
        "days_with_data", "ga4_sessions", "engaged_sessions", "scroll_events",
        "sessions_organic", "sessions_ai", "total_ai_sessions", "days_with_ga4",
        "has_ga4", "word_count", "backlinks", "category_count",
        "search_volume", "competition", "cpc", "content_age_days",
        "days_since_update", "days_since_optimized", "ever_optimized",
        "has_keyword_data",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["word_count"] = df["word_count"].fillna(0)
    df["backlinks"] = df["backlinks"].fillna(0)
    df["category_count"] = df["category_count"].fillna(0)
    df["search_volume"] = df["search_volume"].fillna(0)
    df["competition"] = df["competition"].fillna(0)
    df["cpc"] = df["cpc"].fillna(0)

    for col in ["content_type", "main_intent", "provider_used"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})

    # ═══════════════════════════════════════════════════════════════════════
    # Engineer features — ORDER MATTERS:
    # 1. has_flags first (used by downstream functions)
    # 2. content/market/freshness (base features)
    # 3. ratio/age/visibility (derived from base)
    # 4. position_bucket MUST come before relative_features
    # 5. interaction_features last (uses content_depth, freshness_score, etc.)
    # ═══════════════════════════════════════════════════════════════════════
    df = add_has_flags(df)
    df = add_content_features(df)
    df = add_market_features(df)
    df = add_freshness_features(df)
    df = add_ratio_features(df)
    df = add_age_features(df)
    df = add_visibility_features(df)
    df = add_position_bucket(df)      # <-- BEFORE add_relative_features
    df = add_relative_features(df)    # <-- uses pos_bucket
    df = add_interaction_features(df) # <-- uses content_depth, freshness_score, etc.

    # Save — NO target here (computed downstream to avoid leakage)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Debug
    feature_cols = [c for c in df.columns if c not in ["client_hash_id", "content_hash_id"]]

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Numeric: {sum(1 for c in feature_cols if df[c].dtype.kind in 'iufc')}")
    print(f"Categorical: {sum(1 for c in feature_cols if df[c].dtype.kind == 'O')}")
    print(f"\nSample features:")
    print(df[feature_cols[:10]].head(3).to_string())

    # Metadata
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "initial_rows": int(initial_rows),
        "prepared_rows": int(len(df)),
        "feature_count": len(feature_cols),
        "pos_buckets": POS_LABELS,
    }
    write_json(PROCESSED_DIR / "feature_metadata.json", metadata)


if __name__ == "__main__":
    main()