from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_utils import (
    CHART_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    display_path,
    normalize,
    read_json,
    simple_svg_bar_chart,
    write_json,
)


FEATURE_PATH = PROCESSED_DIR / "refresh_feature_vector.csv"
BASELINE_PATH = PROCESSED_DIR / "baseline_refresh_queue.csv"
PREDICTION_PATH = PROCESSED_DIR / "model_predictions.csv"
MODEL_RESULT_PATH = OUTPUT_DIR / "model_results.json"
QUEUE_PATH = OUTPUT_DIR / "refresh_queue.csv"
REPORT_PATH = OUTPUT_DIR / "model_report.md"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final refresh queue and report.")
    parser.add_argument("--features", default=str(FEATURE_PATH))
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--predictions", default=str(PREDICTION_PATH))
    parser.add_argument("--model-results", default=str(MODEL_RESULT_PATH))
    parser.add_argument("--queue", default=str(QUEUE_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    return parser.parse_args()


# ── Reason codes ─────────────────────────────────────────────────────────────

def merged_reason_codes(row: pd.Series) -> str:
    reasons = [
        reason
        for reason in str(row.get("reason_code", "")).split("+")
        if reason and reason != "nan"
    ]

    if row["best_model_probability"] >= 0.65:
        reasons.append("model_decline_risk")
    if row["best_model_probability"] >= 0.5 and row["total_impressions"] >= 500:
        reasons.append("visible_model_opportunity")
    if (
        row["total_impressions"] >= 500
        and row["avg_position"] > 0
        and row["avg_position"] <= 20
        and row["ctr"] < 0.5
    ):
        reasons.append("ctr_review_candidate")
    if row["ga4_sessions"] >= 30 and (
        (row["engaged_sessions"] > 0 and row["engaged_sessions"] < 30)
        or (row["scroll_events"] > 0 and row["scroll_events"] < 30)
    ):
        reasons.append("engagement_review_candidate")

    unique_reasons = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)
    return "+".join(unique_reasons) if unique_reasons else "general_refresh_review"


def suggested_action(row: pd.Series) -> str:
    reasons = set(str(row["final_reason_codes"]).split("+"))
    if "thin" in reasons:
        return "expand_and_refresh"
    if "ctr_review_candidate" in reasons and (
        "model_decline_risk" in reasons or "declining_with_demand" in reasons
    ):
        return "refresh_and_review_ctr"
    if "engagement_review_candidate" in reasons and (
        "model_decline_risk" in reasons or "declining_with_demand" in reasons
    ):
        return "refresh_and_review_engagement"
    if {
        "model_decline_risk",
        "declining_with_demand",
        "old",
        "visible_model_opportunity",
    }.intersection(reasons):
        return "refresh"
    return "monitor"


def confidence_label(row, high_threshold, medium_threshold):
    score   = row["final_refresh_score"]
    prob    = row["best_model_probability"]
    impr    = row["total_impressions"]
    has_ga4 = row["ga4_sessions"] >= 10


    is_high_standard = (score >= high_threshold and prob >= 0.65 and impr >= 100)
    is_high_ga4_bonus = (score >= high_threshold and prob >= 0.65 and has_ga4)

    if is_high_standard or is_high_ga4_bonus:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"

# ── Charts ───────────────────────────────────────────────────────────────────

def make_charts(final_frame: pd.DataFrame, model_results: dict) -> None:
    action_counts = final_frame["suggested_action"].value_counts().head(10)
    simple_svg_bar_chart(
        "Suggested action mix",
        action_counts.index.tolist(),
        action_counts.values.tolist(),
        CHART_DIR / "action_mix.svg",
        color="#426B69",
    )

    confidence_counts = final_frame["confidence"].value_counts().reindex(
        ["high", "medium", "low"],
        fill_value=0,
    )
    simple_svg_bar_chart(
        "Refresh queue confidence",
        confidence_counts.index.tolist(),
        confidence_counts.values.tolist(),
        CHART_DIR / "confidence_mix.svg",
        color="#6F4E7C",
    )

    reason_counts: dict[str, int] = {}
    for reason_text in final_frame["final_reason_codes"]:
        for reason in str(reason_text).split("+"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    top_reasons = pd.Series(reason_counts).sort_values(ascending=False).head(12)
    simple_svg_bar_chart(
        "Top refresh reason codes",
        top_reasons.index.tolist(),
        top_reasons.values.tolist(),
        CHART_DIR / "top_reason_codes.svg",
        color="#8C6BB1",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Guard: skip feature importance chart if top_features is empty
    # (happens with Logistic Regression Pipeline or Ensemble)
    # ═══════════════════════════════════════════════════════════════════════
    top_features = model_results["best_model"].get("top_features", [])
    if top_features:
        feature_importance = pd.DataFrame(top_features).head(12)
        simple_svg_bar_chart(
            "Top model features",
            feature_importance["feature"].tolist(),
            feature_importance["importance"].tolist(),
            CHART_DIR / "top_feature_importance.svg",
            color="#4E79A7",
        )
    else:
        print("WARNING: No top features available — skipping feature importance chart")

    model_comparison = pd.DataFrame({
        name: {
            "AUC": m["roc_auc"],
            "P@10": m["precision_at_10"],
            "P@20": m["precision_at_20"],
            "P@50": m["precision_at_50"],
            "P@100": m["precision_at_100"],
        }
        for name, m in model_results["models"].items()
    }).T
    simple_svg_bar_chart(
        "Model comparison (P@50)",
        model_comparison.index.tolist(),
        model_comparison["P@50"].tolist(),
        CHART_DIR / "model_comparison.svg",
        color="#B07AA1",
    )
# ── Report ───────────────────────────────────────────────────────────────────

def metric_table(model_results: dict) -> str:
    lines = [
        "| Model | AUC | P@10 | P@20 | P@50 | P@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, metrics in model_results["models"].items():
        lines.append(
            "| "
            + " | ".join([
                model_name,
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['precision_at_10']:.4f}",
                f"{metrics['precision_at_20']:.4f}",
                f"{metrics['precision_at_50']:.4f}",
                f"{metrics['precision_at_100']:.4f}",
            ])
            + " |"
        )
    return "\n".join(lines)


def write_report(final_frame: pd.DataFrame, model_results: dict, report_path: Path) -> None:
    top_features = model_results["best_model"]["top_features"][:10]
    action_counts = final_frame["suggested_action"].value_counts()
    confidence_counts = final_frame["confidence"].value_counts()
    top_preview = final_frame.head(10)

    top_preview_lines = [
        "| Rank | Score | Model probability | Action | Reasons | Impressions | Sessions |",
        "|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in top_preview.itertuples(index=False):
        readable_reasons = str(row.final_reason_codes).replace("+", ", ")
        top_preview_lines.append(
            f"| {row.final_rank} | {row.final_refresh_score:.1f} | "
            f"{row.best_model_probability:.3f} | {row.suggested_action} | "
            f"{readable_reasons} | {int(row.total_impressions)} | "
            f"{int(row.ga4_sessions)} |"
        )

    action_lines = "\n".join(
        f"- `{action}` items: {int(count):,}"
        for action, count in action_counts.sort_values(ascending=False).items()
    )

    report = f"""# FlyRank Refresh Opportunity Model Report

## Data

- Rows scored: {len(final_frame):,}
- Declining-label rows: {int(final_frame["is_below_peer_median"].sum()):,}
- Declining-label rate: {final_frame["is_below_peer_median"].mean():.3f}
- Best model: `{model_results["best_model"]["name"]}`
- Selection metric: `{model_results["best_model"]["selection_metric"]}`

## Model Comparison

{metric_table(model_results)}

## Final Queue

- High-confidence items: {int(confidence_counts.get("high", 0)):,}
- Medium-confidence items: {int(confidence_counts.get("medium", 0)):,}
- Low-confidence items: {int(confidence_counts.get("low", 0)):,}

{action_lines}

## Top Features

"""
    for feature in top_features:
        report += f"- `{feature['feature']}`: {feature['importance']:.6f}\n"

    report += "\n## Top 10 Queue Preview\n\n"
    report += "\n".join(top_preview_lines)
    report += """

## Generated Files

- `outputs/refresh_queue.csv`
- `outputs/model_results.json`
- `outputs/summary.json`
- `outputs/charts/action_mix.svg`
- `outputs/charts/confidence_mix.svg`
- `outputs/charts/top_reason_codes.svg`
- `outputs/charts/top_feature_importance.svg`
- `outputs/charts/model_comparison.svg`

## Practical Use

Use the ranked queue as a reviewer aid, not as an automatic publishing decision.
The safest first production use is to inspect high-confidence rows, verify the page manually,
and compare the recommendation against editorial context.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    feature_frame = pd.read_csv(args.features)
    baseline_frame = pd.read_csv(args.baseline)
    prediction_frame = pd.read_csv(args.predictions)
    model_results = read_json(Path(args.model_results))

    print(f"Feature frame: {feature_frame.shape}")
    print(f"Baseline frame: {baseline_frame.shape}")
    print(f"Prediction frame: {prediction_frame.shape}")

    # Merge baseline + predictions (on content_hash_id)
    final_frame = baseline_frame.merge(
        prediction_frame[
            [
                "content_hash_id",
                "best_model_name",
                "best_model_probability",
            ]
        ],
        on="content_hash_id",
        how="left",
    )

    # Add context columns from feature_frame
    context_columns = [
        "content_hash_id",
        "client_hash_id",
        "total_impressions",
        "ga4_sessions",
        "engaged_sessions",
        "scroll_events",
        "avg_position",
        "ctr",
        "content_age_days",
        "days_since_update",
        "word_count",
        "content_type",
        "main_intent",
        "is_below_peer_median",
    ]
    

    missing_cols = [c for c in context_columns if c not in final_frame.columns]
    if missing_cols:
        print(f"Adding missing columns from feature_frame: {missing_cols}")
        final_frame = final_frame.merge(
            feature_frame[["content_hash_id"] + missing_cols],
            on="content_hash_id",
            how="left",
        )

    print(f"Final frame after merge: {final_frame.shape}")
    print(f"Columns: {list(final_frame.columns)}")

    # Fill missing
    final_frame["best_model_probability"] = final_frame["best_model_probability"].fillna(0)
    final_frame["baseline_score_normalized"] = normalize(final_frame["baseline_score"])

    # Final score = 70% model + 30% baseline
    final_frame["final_refresh_score"] = (
        100 * (
            0.70 * final_frame["best_model_probability"]
            + 0.30 * final_frame["baseline_score_normalized"]
        )
    ).clip(0, 100)

    final_frame["final_reason_codes"] = final_frame.apply(merged_reason_codes, axis=1)
    final_frame["suggested_action"] = final_frame.apply(suggested_action, axis=1)

    high_threshold = float(final_frame["final_refresh_score"].quantile(0.8))
    medium_threshold = float(final_frame["final_refresh_score"].quantile(0.5))
    final_frame["confidence"] = final_frame.apply(
        lambda row: confidence_label(row, high_threshold, medium_threshold),
        axis=1,
    )

    final_frame = final_frame.sort_values(
        ["final_refresh_score", "total_impressions", "ga4_sessions"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    final_frame["final_rank"] = final_frame.index + 1

    # Output columns 
    output_columns = [
        "final_rank",
        "content_hash_id",
        "client_hash_id",
        "final_refresh_score",
        "best_model_name",
        "best_model_probability",
        "baseline_score",
        "confidence",
        "suggested_action",
        "final_reason_codes",
        "is_below_peer_median",
        "total_impressions",
        "ga4_sessions",
        "avg_position",
        "ctr",
        "content_age_days",
        "days_since_update",
        "word_count",
        "content_type",
        "main_intent",
    ]
    
    available_cols = [c for c in output_columns if c in final_frame.columns]
    missing_output = [c for c in output_columns if c not in final_frame.columns]
    if missing_output:
        print(f"WARNING: Missing output columns: {missing_output}")
    
    output_frame = final_frame[available_cols]

    queue_path = Path(args.queue)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(queue_path, index=False)

    # Charts + report
    make_charts(output_frame, model_results)
    write_report(output_frame, model_results, Path(args.report))

    # Summary
    summary_payload = {
        "rows_scored": int(len(output_frame)),
        "best_model": model_results["best_model"]["name"],
        "target_positive_rate": float(model_results["best_model"]["metrics"]["precision"]),
        "final_score_p80": high_threshold,
        "final_score_p50": medium_threshold,
        "top_queue_score": float(output_frame["final_refresh_score"].max()),
        "high_confidence_rows": int((output_frame["confidence"] == "high").sum()),
        "queue_output": display_path(queue_path),
        "report_output": display_path(args.report),
        "charts": [
            display_path(CHART_DIR / "action_mix.svg"),
            display_path(CHART_DIR / "confidence_mix.svg"),
            display_path(CHART_DIR / "top_reason_codes.svg"),
            display_path(CHART_DIR / "top_feature_importance.svg"),
            display_path(CHART_DIR / "model_comparison.svg"),
        ],
    }
        # Load baseline metrics for comparison
    baseline_metadata_path = PROCESSED_DIR / "baseline_metadata.json"
    if baseline_metadata_path.exists():
        baseline_meta = read_json(baseline_metadata_path)
        if "metrics" in baseline_meta:
            summary_payload["baseline"] = baseline_meta["metrics"]
            summary_payload["baseline"]["thresholds"] = baseline_meta.get("thresholds", {})
    write_json(SUMMARY_PATH, summary_payload)

    print(f"{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Queue: {queue_path}")
    print(f"Report: {args.report}")
    print(f"Charts: {CHART_DIR}")
    print(f"High confidence: {summary_payload['high_confidence_rows']:,} rows")
    

if __name__ == "__main__":
    main()
    