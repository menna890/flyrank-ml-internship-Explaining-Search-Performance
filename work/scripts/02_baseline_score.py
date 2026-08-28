from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_utils import (
    PROCESSED_DIR,
    ensure_dirs,
    precision_at_k,
    write_json,
)


FEATURE_PATH = PROCESSED_DIR / "refresh_feature_vector.csv"
OUTPUT_PATH = PROCESSED_DIR / "baseline_refresh_queue.csv"

# ── Drop list (leakage columns — same as model) ──────────────────────────────

DROP_COLS = [
    'client_hash_id', 'content_hash_id',
    'is_underperforming', 'underperformance_score',
    'expected_clicks', 'expected_ctr',
    'is_below_peer_median', 'pos_bucket',
    'ctr', 'total_clicks',
    'ga4_sessions', 'log_ga4_sessions',
    'engaged_sessions', 'scroll_events',
    'sessions_organic', 'sessions_direct',
    'sessions_referral', 'sessions_social',
    'sessions_paid', 'sessions_ai',
    'total_ai_sessions', 'days_with_ga4',
    'has_ga4','avg_position',
    'total_impressions',
'impressions_per_day', 'lifetime_impressions_est',
'is_high_visibility', 'market_capture_ratio',
'cat_impression_ratio'
]


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic refresh baseline score.")
    parser.add_argument("--input", default=str(FEATURE_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    return parser.parse_args()


# ── Target computation (train-only median CTR) ───────────────────────────────

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute is_below_peer_median using full-data median CTR per pos_bucket."""
    # For baseline, we use full data (not train-only) since it's a rule, not a model
    bucket_stats = (
        df.groupby("pos_bucket", observed=True)["ctr"]
        .median()
        .to_dict()
    )

    print(f"\nMedian CTR per position bucket:")
    for bucket, median_ctr in sorted(bucket_stats.items()):
        print(f"  {bucket}: {median_ctr:.5f}")

    df["expected_ctr"] = df["pos_bucket"].map(bucket_stats)
    df["is_below_peer_median"] = (df["ctr"] < df["expected_ctr"]).astype(int)

    print(f"\nTarget distribution:")
    print(df["is_below_peer_median"].value_counts().to_string())

    return df


# ── Baseline scoring ─────────────────────────────────────────────────────────

def make_reason(row: pd.Series) -> str:
    reasons = []
    if row["rule_old"]:
        reasons.append("old")
    if row["rule_never_opt"]:
        reasons.append("never_optimized")
    if row["rule_thin"]:
        reasons.append("thin")
    if row["rule_no_keyword"]:
        reasons.append("no_keyword")
    return "+".join(reasons) if reasons else "no_signal"


def main() -> None:
    args = parse_args()
    ensure_dirs()

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Feature vector is empty")

    print(f"{'='*60}")
    print("BASELINE SCORE — Rule-based")
    print(f"{'='*60}")

    # Compute target
    df = compute_target(df)

    # Thresholds from data
    age_median = df["content_age_days"].median()
    wc_median = df["word_count"].median()

    print(f"\nThresholds:")
    print(f"  age_median : {age_median:.0f} days")
    print(f"  wc_median  : {wc_median:.0f} words")

    # Rules (on features AFTER drop — but these are computed from non-leaky cols)
    df["rule_old"] = (df["content_age_days"] > age_median).astype(int)
    df["rule_never_opt"] = (df["ever_optimized"] == 0).astype(int)
    df["rule_thin"] = (df["word_count"] < wc_median).astype(int)
    df["rule_no_keyword"] = (df["has_keyword_data"] == 0).astype(int)

    # Baseline score
    df["baseline_score"] = (
        df["rule_old"] +
        df["rule_never_opt"] +
        df["rule_thin"] +
        df["rule_no_keyword"]
    )

    # Reason code
    df["reason_code"] = df.apply(make_reason, axis=1)

    # Rank
    df["baseline_rank"] = df["baseline_score"].rank(method="first", ascending=False).astype(int)

    # ── Evaluate vs target ─────────────────────────────────────────────────
    y_true = df["is_below_peer_median"].values
    baseline_scr = df["baseline_score"].values
    random_scr = np.random.RandomState(42).rand(len(y_true))

    print(f"\n{'─'*45}")
    print(f"Base rate: {y_true.mean()*100:.1f}%")
    print(f"\n{'Method':<25} {'P@10':>6} {'P@20':>6} {'P@50':>6}")
    print(f"{'─'*45}")

    for name, scores in [("Random", random_scr), ("Baseline Rule", baseline_scr)]:
        p10 = precision_at_k(y_true, scores, 10)
        p20 = precision_at_k(y_true, scores, 20)
        p50 = precision_at_k(y_true, scores, 50)
        print(f"{name:<25} {p10:>6.3f} {p20:>6.3f} {p50:>6.3f}")

    # Score distribution
    print(f"\nBaseline score distribution:")
    print(df["baseline_score"].value_counts().sort_index().to_string())

    print(f"\nGap rate per score:")
    print(
        df.groupby("baseline_score")["is_below_peer_median"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "gap_rate", "count": "n"})
        .round(3)
        .to_string()
    )

    # Top 20 review
    print(f"\nTop 20 (score=4 pages):")
    top20 = df[df["baseline_score"] == 4][[
        "content_hash_id", "content_age_days", "ever_optimized",
        "word_count", "has_keyword_data", "baseline_score", "reason_code",
        "is_below_peer_median",
    ]].head(20)
    print(top20.to_string(index=False))
    correct = top20["is_below_peer_median"].mean()
    print(f"\nTop 20 precision: {correct:.2f}")
    print(f"FP in top 20: {int((1-correct)*20)}/20")

    # Save — keep target for reference but drop leakage cols from features
    output_cols = [
        "content_hash_id",
        "client_hash_id",
        "baseline_rank",
        "baseline_score",
        "reason_code",
        "rule_old",
        "rule_never_opt",
        "rule_thin",
        "rule_no_keyword",
        "is_below_peer_median",  # ← keep for evaluation
        "content_age_days",
        "ever_optimized",
        "word_count",
        "has_keyword_data",
    ]

    out = df[output_cols].sort_values("baseline_rank")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    # Metadata
    metadata = {
        "rows": int(len(out)),
        "top_score": int(df["baseline_score"].max()),
        "median_score": float(df["baseline_score"].median()),
        "score_distribution": df["baseline_score"].value_counts().to_dict(),
        "thresholds": {
            "age_median": float(age_median),
            "wc_median": float(wc_median),
        },
        "precision_at_50": precision_at_k(y_true, baseline_scr, 50),
    }
    # Calculate all metrics for comparison
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    baseline_binary = (baseline_scr >= 2).astype(int)

    metadata["metrics"] = {
    "accuracy": float(accuracy_score(y_true, baseline_binary)),
    "precision": float(precision_score(y_true, baseline_binary, zero_division=0)),
    "recall": float(recall_score(y_true, baseline_binary, zero_division=0)),
    "f1": float(f1_score(y_true, baseline_binary, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_true, baseline_scr)) if len(set(y_true)) == 2 else 0.0,
    "average_precision": float(average_precision_score(y_true, baseline_scr)) if len(set(y_true)) == 2 else 0.0,
    "precision_at_10": precision_at_k(y_true, baseline_scr, 10),
    "precision_at_20": precision_at_k(y_true, baseline_scr, 20),
    "precision_at_50": precision_at_k(y_true, baseline_scr, 50),
    "precision_at_100": precision_at_k(y_true, baseline_scr, 100),
    }
    write_json(PROCESSED_DIR / "baseline_metadata.json", metadata)

    print(f"\n{'='*60}")
    print(f"Wrote baseline queue: {output_path}")
    print(f"Rows: {len(out):,}")


if __name__ == "__main__":
    main()