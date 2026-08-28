# FlyRank Refresh Opportunity Model Report

## Data

- Rows scored: 118,092
- Declining-label rows: 45,434
- Declining-label rate: 0.385
- Best model: `CatBoost`
- Selection metric: `roc_auc`

## Model Comparison

| Model | AUC | P@10 | P@20 | P@50 | P@100 |
|---|---:|---:|---:|---:|---:|
| CatBoost | 0.8505 | 0.8000 | 0.8000 | 0.8200 | 0.8500 |
| LightGBM | 0.8498 | 0.9000 | 0.8500 | 0.8200 | 0.8300 |
| Logistic Regression | 0.7835 | 0.9000 | 0.6500 | 0.7400 | 0.7300 |
| Random Forest | 0.8254 | 0.9000 | 0.8000 | 0.7200 | 0.6700 |
| XGBoost | 0.8478 | 0.7000 | 0.7500 | 0.8600 | 0.8300 |

## Final Queue

- High-confidence items: 16,806
- Medium-confidence items: 42,240
- Low-confidence items: 59,046

- `expand_and_refresh` items: 59,000
- `monitor` items: 27,191
- `refresh` items: 26,900
- `refresh_and_review_ctr` items: 4,982
- `refresh_and_review_engagement` items: 19

## Top Features

- `search_volume_vs_bucket_mean`: 16.486834
- `content_age_days_vs_bucket_median`: 15.080703
- `word_count_vs_bucket_median`: 12.741437
- `search_volume_vs_bucket_median`: 5.844059
- `days_since_update_vs_bucket_mean`: 5.603688
- `log_age`: 5.515716
- `age_squared`: 4.901371
- `content_age_days`: 4.414785
- `data_completeness`: 3.028298
- `content_age_days_vs_bucket_mean`: 2.629106

## Top 10 Queue Preview

| Rank | Score | Model probability | Action | Reasons | Impressions | Sessions |
|---:|---:|---:|---|---|---:|---:|
| 1 | 91.0 | 0.871 | expand_and_refresh | old, never_optimized, thin, no_keyword, model_decline_risk | 102 | 0 |
| 2 | 90.6 | 0.973 | refresh | old, never_optimized, no_keyword, model_decline_risk | 101 | 0 |
| 3 | 90.3 | 0.969 | refresh | old, never_optimized, no_keyword, model_decline_risk | 157 | 0 |
| 4 | 90.3 | 0.969 | refresh | old, never_optimized, no_keyword, model_decline_risk | 135 | 0 |
| 5 | 90.2 | 0.860 | expand_and_refresh | old, never_optimized, thin, no_keyword, model_decline_risk | 137 | 0 |
| 6 | 90.1 | 0.858 | expand_and_refresh | old, never_optimized, thin, no_keyword, model_decline_risk, visible_model_opportunity, ctr_review_candidate | 2002 | 2 |
| 7 | 89.5 | 0.957 | expand_and_refresh | old, never_optimized, thin, model_decline_risk, visible_model_opportunity, ctr_review_candidate | 1223 | 3 |
| 8 | 89.2 | 0.846 | expand_and_refresh | old, never_optimized, thin, no_keyword, model_decline_risk | 106 | 0 |
| 9 | 89.2 | 0.952 | expand_and_refresh | old, never_optimized, thin, model_decline_risk, visible_model_opportunity, ctr_review_candidate | 1377 | 0 |
| 10 | 89.0 | 0.843 | expand_and_refresh | old, never_optimized, thin, no_keyword, model_decline_risk, visible_model_opportunity, ctr_review_candidate | 501 | 0 |

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
