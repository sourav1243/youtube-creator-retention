# Model Card: K-Means Clustering (v1)

Generated: 2026-06-19T09:48:18.576428+00:00

## Model Overview

- Algorithm: K-Means
- Number of clusters (K): 2
- Silhouette score: 0.3516
- Random state: 42
- Feature columns: upload_freq_30d, upload_freq_90d, freq_trend_ratio, momentum_ratio, avg_engagement_rate, days_since_last_upload
- Scaler: RobustScaler (median/IQR — justified: YouTube metrics are outlier-heavy)

## Cluster Sizes & Labels

| Cluster | Label | Risk Flag | Size |
|---|---|---|---|
| 0 | Low Momentum — Declining Engagement | At-Risk | 82 |
| 1 | High Momentum — Frequent Uploaders | Healthy | 13 |

## Centroid Values (unscaled)

| Feature | 0 | 1 |
|---|---|---|
| upload_freq_30d | 0.2433 | 0.5275 |
| upload_freq_90d | 0.2590 | 0.4416 |
| freq_trend_ratio | 2.7802 | 2.2180 |
| momentum_ratio | 1.1189 | 1.4311 |
| avg_engagement_rate | 0.0854 | 0.0758 |
| days_since_last_upload | 6.0854 | 31.2308 |

## Labeling Rule

- Clusters are ranked by combined (momentum_ratio + upload_freq_30d) centroid values.
- Lowest-ranked cluster → 'Low Momentum — Declining Engagement' → At-Risk
- Mid-ranked cluster(s) → 'Mid Momentum — Regular Creators' → Watch
- Highest-ranked cluster → 'High Momentum — Frequent Uploaders' → Healthy
- Channels with insufficient_history → Unscored (not included in clustering)
