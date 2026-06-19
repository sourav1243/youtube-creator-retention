# Model Card: KMEANS Clustering

Generated: 2026-06-19T16:42:24.983117+00:00

## Model Overview

- Algorithm: KMEANS
- Number of clusters (K): 3
- Silhouette score: 0.8065
- Bootstrap stability (mean confidence): 0.7845
- Random state: 42
- Feature columns: upload_freq_30d, upload_freq_90d, freq_trend_ratio, momentum_ratio, avg_engagement_rate, days_since_last_upload, upload_regularity
- Preprocessing: RobustScaler on continuous features, log1p on days_since_last_upload

## Risk Score Formula

A direct feature-based risk score is computed independently of clustering:

  - `avg_engagement_rate`: 15%
  - `days_since_last_upload`: 15%
  - `freq_trend_ratio`: 10%
  - `momentum_ratio`: 30%
  - `upload_freq_30d`: 20%
  - `upload_regularity`: 10%

Each component is normalized to [0,1] and weighted. Higher score = higher churn risk.

## Cluster Sizes & Labels

| Cluster | Label | Risk Flag | Size | Mean Risk Score |
|---|---|---|---|---|
| 0 | High Momentum — Frequent Uploaders | At-Risk | 480 | 0.3711 |
| 1 | Low Momentum — Declining Engagement | At-Risk | 11 | 0.6396 |
| 2 | Mid Momentum — Regular Creators | Watch | 1 | 0.3817 |

## Centroid Values (unscaled)

| Feature | 0 | 1 | 2 |
|---|---|---|---|
| upload_freq_30d | 1.1977 | 0.0394 | 0.0667 |
| upload_freq_90d | 0.7001 | 0.0566 | 0.0444 |
| freq_trend_ratio | 1.3794 | 1.0121 | 1.5000 |
| momentum_ratio | 5.1888 | 1.9318 | 407.4753 |
| avg_engagement_rate | 0.0376 | 0.0215 | 0.0379 |
| days_since_last_upload | 5.7812 | 26.1818 | 1.0000 |
| upload_regularity | 5.0090 | 129.0288 | 20.7834 |

## Labeling Rule

- Clusters are ranked by mean risk_score of their members.
- Lowest-risk cluster(s) → Healthy
- Mid-risk cluster(s) → Watch
- Highest-risk cluster(s) → At-Risk
- Channels with insufficient_history → Unscored (not included in clustering)

## Risk Score Weights (JSON)

```json
{
  "upload_freq_30d": 0.2,
  "freq_trend_ratio": 0.1,
  "momentum_ratio": 0.3,
  "days_since_last_upload": 0.15,
  "avg_engagement_rate": 0.15,
  "upload_regularity": 0.1
}
```
