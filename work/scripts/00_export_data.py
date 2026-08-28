from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd

from ml_utils import RAW_PATH, ensure_dirs


# ── Config ───────────────────────────────────────────────────────────────────

START_DATE_3M = "2026-01-01"
END_DATE_3M = "2026-03-31"
REF_DATE_3M = "2026-03-31"

START_DATE_APR = "2026-04-01"
END_DATE_APR = "2026-04-30"
REF_DATE_APR = "2026-04-30"

# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw data from DuckDB + HuggingFace to Parquet.")
    parser.add_argument(
        "--output-3m",
        default=str(RAW_PATH.with_name("content_refresh_3m.parquet")),
        help="Where to write the 3-month Parquet file.",
    )
    parser.add_argument(
        "--output-apr",
        default=str(RAW_PATH.with_name("content_refresh_apr.parquet")),
        help="Where to write the April Parquet file.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var / .env file).",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_hf_token(args_token: str | None) -> str:
    if args_token:
        return args_token
    
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    

    
    raise ValueError(
        "HF_TOKEN is required. Pass --hf-token, set HF_TOKEN env var, or add to .env file."
    )


def build_query(start_date: str, end_date: str, ref_date: str) -> str:
    return f"""
    WITH filtered_daily AS (
        SELECT *
        FROM 'hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month=2026-0*/*.parquet'
        WHERE gsc_avg_position != 0
          AND report_date >= '{start_date}'
          AND report_date <= '{end_date}'
    ),
    page_aggregates AS (
        SELECT
            client_hash_id,
            content_hash_id,
            AVG(gsc_avg_position) AS avg_position,
            SUM(gsc_clicks) AS total_clicks,
            SUM(gsc_impressions) AS total_impressions,
            (SUM(gsc_clicks) * 1.0 / NULLIF(SUM(gsc_impressions), 0)) AS ctr,
            COUNT(DISTINCT report_date) AS days_with_data,
            SUM(COALESCE(ga4_sessions, 0)) AS ga4_sessions,
            SUM(COALESCE(ga4_engaged_sessions, 0)) AS engaged_sessions,
            SUM(COALESCE(scroll_events, 0)) AS scroll_events,
            SUM(COALESCE(sessions_organic, 0)) AS sessions_organic,
            SUM(COALESCE(sessions_ai, 0)) AS sessions_ai,
            SUM(
                COALESCE(ai_chatgpt, 0) + COALESCE(ai_perplexity, 0) +
                COALESCE(ai_gemini, 0) + COALESCE(ai_copilot, 0) +
                COALESCE(ai_claude, 0) + COALESCE(ai_meta, 0) + COALESCE(ai_other, 0)
            ) AS total_ai_sessions,
            SUM(CASE WHEN ga4_data_available IS TRUE THEN 1 ELSE 0 END) AS days_with_ga4,
            CASE WHEN SUM(CASE WHEN ga4_data_available IS TRUE THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_ga4
        FROM filtered_daily
        GROUP BY client_hash_id, content_hash_id
    ),
    safe_dim_content AS (
        SELECT *
        FROM 'hf://datasets/FlyRank/internship-warehouse/dim_content.parquet'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY content_hash_id ORDER BY content_hash_id) = 1
    )
    SELECT
        f.*,
        COALESCE(c.word_count, 0) AS word_count,
        c.content_type,
        COALESCE(c.backlinks, 0) AS backlinks,
        COALESCE(c.category_count, 0) AS category_count,
        COALESCE(c.search_volume, 0) AS search_volume,
        COALESCE(c.competition, 0.0) AS competition,
        COALESCE(c.cpc, 0.0) AS cpc,
        c.main_intent,
        c.provider_used,
        DATEDIFF('day', c.content_created_date, DATE '{ref_date}') AS content_age_days,
        DATEDIFF('day', c.content_updated_date, DATE '{ref_date}') AS days_since_update,
        CASE WHEN c.last_optimized_date IS NOT NULL
             THEN DATEDIFF('day', c.last_optimized_date, DATE '{ref_date}')
             ELSE -1 END AS days_since_optimized,
        CASE WHEN c.last_optimized_date IS NOT NULL THEN 1 ELSE 0 END AS ever_optimized,
        CASE WHEN c.search_volume IS NOT NULL THEN 1 ELSE 0 END AS has_keyword_data
    FROM page_aggregates f
    LEFT JOIN safe_dim_content c USING (content_hash_id);
    """


def run_export(con: duckdb.DuckDBPyConnection, label: str, start: str, end: str, ref: str, output_path: Path) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"▶ Exporting: {label}")
    print(f"   Window: [{start} to {end}]")
    print(f"   Ref date: {ref}")
    print(f"   Output: {output_path}")

    query = build_query(start, end, ref)
    df = con.execute(query).df()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # ── Debug ──────────────────────────────────────────────────────────────
    print(f"\n   ✓ Saved: {output_path}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   ── Sample (first 3 rows) ──")
    print(df.head(3).to_string())
    print(f"\n   ── Numeric summary ──")
    print(df.describe().transpose().to_string())
    print(f"\n   ── Null counts ──")
    print(df.isnull().sum().to_string())
    print(f"\n   ── Categorical value counts ──")
    for col in ["content_type", "main_intent", "provider_used"]:
        if col in df.columns:
            print(f"\n   {col}:")
            print(df[col].value_counts().head(10).to_string())

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ensure_dirs()

    # Setup DuckDB + HuggingFace auth
    con = duckdb.connect()
    hf_token = get_hf_token(args.hf_token)
    con.execute(f"CREATE SECRET (TYPE huggingface, TOKEN '{hf_token}')")

    # Export 3-month window
    df_3m = run_export(
        con, "3-Month Window (Jan–Mar)",
        START_DATE_3M, END_DATE_3M, REF_DATE_3M,
        Path(args.output_3m)
    )

    # Export April window
    df_apr = run_export(
        con, "April Window",
        START_DATE_APR, END_DATE_APR, REF_DATE_APR,
        Path(args.output_apr)
    )

    # Final summary
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"  3-Month:  {df_3m.shape[0]:,} rows → {args.output_3m}")
    print(f"  April:    {df_apr.shape[0]:,} rows → {args.output_apr}")


if __name__ == "__main__":
    main()