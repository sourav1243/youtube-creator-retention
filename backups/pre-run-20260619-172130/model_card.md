# Model Card: K-Means Clustering (v1)

Generated: 2026-06-19T11:21:11.974257+00:00

## Model Overview

- Algorithm: K-Means
- Number of clusters (K): 2
- Silhouette score: 0.9075
- Random state: 42
- Feature columns: upload_freq_30d, upload_freq_90d, freq_trend_ratio, momentum_ratio, avg_engagement_rate, days_since_last_upload
- Scaler: RobustScaler (median/IQR — justified: YouTube metrics are outlier-heavy)

## Cluster Sizes & Labels

| Cluster | Label | Risk Flag | Size |
|---|---|---|---|
| 0 | High Momentum — Frequent Uploaders | Healthy | 33 |
| 1 | Low Momentum — Declining Engagement | At-Risk | 2 |

## Centroid Values (unscaled)

| Feature | 0 | 1 |
|---|---|---|
| upload_freq_30d | 2.0192 | 0.0000 |
| upload_freq_90d | 0.9859 | 0.2611 |
| freq_trend_ratio | 1.6814 | 0.0000 |
| momentum_ratio | 3.8883 | 2.1321 |
| avg_engagement_rate | 0.0285 | 0.0557 |
| days_since_last_upload | 0.9091 | 45.0000 |

## Labeling Rule

- Clusters are ranked by combined (momentum_ratio + upload_freq_30d) centroid values.
- Lowest-ranked cluster → 'Low Momentum — Declining Engagement' → At-Risk
- Mid-ranked cluster(s) → 'Mid Momentum — Regular Creators' → Watch
- Highest-ranked cluster → 'High Momentum — Frequent Uploaders' → Healthy
- Channels with insufficient_history → Unscored (not included in clustering)
