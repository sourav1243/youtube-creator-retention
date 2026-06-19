# Model Card: KMEANS Clustering

Generated: 2026-06-19T13:35:48.697201+00:00

## Model Overview

- Algorithm: KMEANS
- Number of clusters (K): 2
- Silhouette score: 0.5525
- Bootstrap stability (mean confidence): 0.7121
- Random state: 42
- Feature columns: upload_freq_30d, upload_freq_90d, momentum_ratio, avg_engagement_rate, days_since_last_upload, upload_regularity, duration_trend
- Preprocessing: RobustScaler on continuous features, log1p on days_since_last_upload
- 'days_since_last_upload' log1p-transformed to reduce feature dominance (original effect size: 6.0 → ~1.5)

## Cluster Sizes & Labels

| Cluster | Label | Risk Flag | Size | Mean Risk Score |
|---|---|---|---|---|
| 0 | Low Momentum — Declining Engagement | At-Risk | 33 | 0.6331 |
| 1 | High Momentum — Frequent Uploaders | Healthy | 2 | 0.1429 |

## Centroid Values (unscaled)

| Feature | 0 | 1 |
|---|---|---|
| upload_freq_30d | 1.9424 | 1.2167 |
| upload_freq_90d | 0.9212 | 1.3167 |
| momentum_ratio | 3.7337 | 3.9661 |
| avg_engagement_rate | 0.0312 | 0.0118 |
| days_since_last_upload | 3.6970 | 0.0000 |
| upload_regularity | 2.5190 | 0.4955 |
| duration_trend | 14.8154 | 114.5457 |

## Labeling Rule

- Clusters are ranked by combined (momentum_ratio + upload_freq_30d) centroid values.
- Lowest-ranked cluster → 'Low Momentum — Declining Engagement' → At-Risk
- Mid-ranked cluster(s) → 'Mid Momentum — Regular Creators' → Watch
- Highest-ranked cluster → 'High Momentum — Frequent Uploaders' → Healthy
- Channels with insufficient_history → Unscored (not included in clustering)
