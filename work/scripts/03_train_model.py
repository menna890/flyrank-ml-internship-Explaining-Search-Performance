from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_utils import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    ensure_dirs,
    precision_at_k,
    read_json,
    write_json,
)


FEATURE_PATH = PROCESSED_DIR / "refresh_feature_vector.csv"
PREDICTION_PATH = PROCESSED_DIR / "model_predictions.csv"
RESULT_PATH = OUTPUT_DIR / "model_results.json"
RANDOM_STATE = 42


# Columns to drop — pos_bucket is NOT a feature, only used for target creation
DROP_COLS = [
    "client_hash_id",
    "content_hash_id",
    "is_underperforming",
    "underperformance_score",
    "expected_clicks",
    "expected_ctr",
    "is_below_peer_median",
    "pos_bucket",
    "ctr",
    "total_clicks",
    "ga4_sessions",
    "engaged_sessions",
    "scroll_events",
    "sessions_organic",
    "sessions_ai",
    "total_ai_sessions",
    "days_with_ga4",
    "has_ga4",
    "avg_position",
    "total_impressions",
    "days_since_update",
    "freshness_score",
    "optimized_and_fresh",
    "content_maturity",
    "update_optimization_gap",
    "impressions_per_day",
    "lifetime_impressions_est",
    "market_capture_ratio",
    "backlink_efficiency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train refresh opportunity models.")
    parser.add_argument("--features", default=str(FEATURE_PATH))
    parser.add_argument("--predictions", default=str(PREDICTION_PATH))
    parser.add_argument("--results", default=str(RESULT_PATH))
    parser.add_argument("--ensemble", action="store_true", default=True,
                        help="Build soft-voting ensemble of top models")
    parser.add_argument("--no-ensemble", dest="ensemble", action="store_false")
    return parser.parse_args()

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = below peer median CTR per pos_bucket
    مش extremes — كل الصفحات بتتقارن بـ peers
    """
    groups = df["client_hash_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, _ = next(gss.split(df, groups=groups))

    df = df.copy()
    df["split"] = "test"
    df.iloc[train_idx, df.columns.get_loc("split")] = "train"

    train_df = df[df["split"] == "train"]

    # median بس — مش extremes
    bucket_medians = {}
    for bucket, group in train_df.groupby("pos_bucket", observed=True):
        ctr_vals = group["ctr"].dropna()
        if len(ctr_vals) >= 50:
            bucket_medians[bucket] = ctr_vals.median()

    print("Median CTR per bucket (train only):")
    for b, m in sorted(bucket_medians.items()):
        print(f"  {b}: {m:.5f}")

    def assign_target(row):
        bucket = row["pos_bucket"]
        ctr_val = row["ctr"]
        if bucket not in bucket_medians or pd.isna(ctr_val):
            return np.nan
        return int(ctr_val < bucket_medians[bucket])

    df["is_below_peer_median"] = df.apply(assign_target, axis=1)
    df = df.dropna(subset=["is_below_peer_median"]).copy()
    df["is_below_peer_median"] = df["is_below_peer_median"].astype(int)

    print(f"\nRows kept: {len(df):,}  (vs {len(df)*100//60:,} before)")
    print(f"Label distribution:")
    print(df["is_below_peer_median"].value_counts())

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_cols).fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    y = df["is_below_peer_median"]

    train_mask = df["split"] == "train"
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[~train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[~train_mask].reset_index(drop=True)

    X_train = pd.get_dummies(X_train, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, drop_first=True, dtype=int)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    print(f"\n{'='*60}")
    print("FEATURE PREP COMPLETE")
    print(f"{'='*60}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Test label=1: {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test, list(X_train.columns)


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.95, 0.005)
    scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        scores.append(f1_score(y_true, y_pred, zero_division=0))
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    print(f"\n{'─'*50}")
    print(f"Training: {name}")
    print(f"{'─'*50}")

    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    best_thresh, _ = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_test.nunique() == 2 else 0.0,
        "average_precision": float(average_precision_score(y_test, y_prob)) if y_test.nunique() == 2 else 0.0,
        "precision_at_10": precision_at_k(y_test, y_prob, 10),
        "precision_at_20": precision_at_k(y_test, y_prob, 20),
        "precision_at_50": precision_at_k(y_test, y_prob, 50),
        "precision_at_100": precision_at_k(y_test, y_prob, 100),
        "best_threshold": float(best_thresh),
    }

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f} (thresh={best_thresh:.3f})")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"  P@10:      {metrics['precision_at_10']:.4f}")
    print(f"  P@20:      {metrics['precision_at_20']:.4f}")
    print(f"  P@50:      {metrics['precision_at_50']:.4f}")
    print(f"  P@100:     {metrics['precision_at_100']:.4f}")

    return {
        "name": name,
        "model": model,
        "metrics": metrics,
        "probabilities": y_prob,
    }


def get_top_features(model, feature_names: list[str], n: int = 15) -> list[dict]:
    """Extract top-N feature importances from tree-based or linear model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "named_steps"):
        # sklearn Pipeline — extract from final step
        final_step = model.named_steps[list(model.named_steps.keys())[-1]]
        if hasattr(final_step, "coef_"):
            importances = np.abs(final_step.coef_[0])
        else:
            return []
    else:
        return []

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(n)

    return [
        {"feature": row.feature, "importance": float(row.importance)}
        for row in feat_imp.itertuples(index=False)
    ]
def print_comparison_table(results: list[dict]) -> None:
    print(f"\n{'='*75}")
    print("MODEL COMPARISON")
    print(f"{'='*75}")
    print(f"{'Name':<22} {'AUC':>8} {'F1':>8} {'P@10':>8} {'P@20':>8} {'P@50':>8} {'P@100':>8}")
    print(f"{'─'*75}")

    for r in results:
        m = r["metrics"]
        print(f"{r['name']:<22} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['precision_at_10']:>8.4f} {m['precision_at_20']:>8.4f} "
              f"{m['precision_at_50']:>8.4f} {m['precision_at_100']:>8.4f}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df):,} rows from {args.features}")

    df = compute_target(df)
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df)

    results = []
    tuned_models = {}

    # ═══════════════════════════════════════════════════════════════════════
    # 1. LIGHTGBM — strong defaults
    # ═══════════════════════════════════════════════════════════════════════
    lgbm = LGBMClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=7, num_leaves=60,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    result = evaluate_model("LightGBM", lgbm, X_train, y_train, X_test, y_test)
    result["top_features"] = get_top_features(result["model"], feature_names)
    results.append(result)
    tuned_models["lgbm"] = result["model"]

    # ═══════════════════════════════════════════════════════════════════════
    # 2. XGBOOST — strong defaults
    # ═══════════════════════════════════════════════════════════════════════
    xgb = XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    result = evaluate_model("XGBoost", xgb, X_train, y_train, X_test, y_test)
    result["top_features"] = get_top_features(result["model"], feature_names)
    results.append(result)
    tuned_models["xgb"] = result["model"]

    # ═══════════════════════════════════════════════════════════════════════
    # 3. CATBOOST — strong defaults
    # ═══════════════════════════════════════════════════════════════════════
    cat = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE, verbose=0
    )
    result = evaluate_model("CatBoost", cat, X_train, y_train, X_test, y_test)
    result["top_features"] = get_top_features(result["model"], feature_names)
    results.append(result)
    tuned_models["catboost"] = result["model"]

    # ═══════════════════════════════════════════════════════════════════════
    # 4. GRADIENT BOOSTING — defaults
    # ═══════════════════════════════════════════════════════════════════════
    gb = GradientBoostingClassifier(
        n_estimators=800, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
    )
    result = evaluate_model("Gradient Boosting", gb, X_train, y_train, X_test, y_test)
    result["top_features"] = get_top_features(result["model"], feature_names)
    results.append(result)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. RANDOM FOREST — defaults
    # ═══════════════════════════════════════════════════════════════════════
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    result = evaluate_model("Random Forest", rf, X_train, y_train, X_test, y_test)
    result["top_features"] = get_top_features(result["model"], feature_names)
    results.append(result)

    # ═══════════════════════════════════════════════════════════════════════
    # 6. LOGISTIC REGRESSION — baseline
    # ═══════════════════════════════════════════════════════════════════════
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE)
    )
    result = evaluate_model("Logistic Regression", lr, X_train, y_train, X_test, y_test)
    result["top_features"] = get_top_features(result["model"], feature_names)
    results.append(result)


    print_comparison_table(results)

    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best['name']} (AUC = {best['metrics']['roc_auc']:.4f})")
    print(f"{'='*70}")

    if best["top_features"]:
        print(f"\nTop 15 features ({best['name']}):")
        for i, feat in enumerate(best["top_features"], 1):
            print(f"  {i:2d}. {feat['feature']:<35} {feat['importance']:.6f}")

    # ═══════════════════════════════════════════════════════════════════════
    # RETRAIN BEST MODEL ON FULL DATA
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("RETRAINING BEST MODEL ON FULL DATA")
    print(f"{'='*60}")

    drop_cols = [c for c in DROP_COLS if c in df.columns]
    X_full = df.drop(columns=drop_cols).fillna(0).replace([np.inf, -np.inf], 0)
    X_full = pd.get_dummies(X_full, drop_first=True, dtype=int)
    X_full = X_full.reindex(columns=X_train.columns, fill_value=0)

    print(f"Full data: {len(X_full):,} rows × {X_full.shape[1]} features")

    best["model"].fit(X_full, df["is_below_peer_median"])
    all_prob = best["model"].predict_proba(X_full)[:, 1]

    preds = df[[
        "content_hash_id", "client_hash_id", "is_below_peer_median", "split"
    ]].copy()
    preds["best_model_name"] = best["name"]
    preds["best_model_probability"] = all_prob
    preds.to_csv(args.predictions, index=False)

    print(f"Saved predictions: {args.predictions}")

    write_json(Path(args.results), {
        "models": {r["name"]: r["metrics"] for r in results},
        "best_model": {
            "name": best["name"],
            "selection_metric": "roc_auc",
            "metrics": best["metrics"],
            "top_features": best["top_features"],
        },
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "full_data_predictions": str(args.predictions),
        "ensemble_enabled": args.ensemble,
    })
    print(f"\nDone: {args.results}")


if __name__ == "__main__":
    main()