# Model Card: KMEANS Clustering

Generated: 2026-06-19T15:11:45.589316+00:00

## Model Overview

- Algorithm: KMEANS
- Number of clusters (K): 3
- Silhouette score: 0.5134
- Bootstrap stability (mean confidence): 0.5499
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
| 0 | High Momentum — Frequent Uploaders | Healthy | 30 | 0.3103 |
| 1 | Mid Momentum — Regular Creators | At-Risk | 1 | 0.4002 |
| 2 | Low Momentum — Declining Engagement | At-Risk | 4 | 0.4700 |

## Centroid Values (unscaled)

| Feature | 0 | 1 | 2 |
|---|---|---|---|
| upload_freq_30d | 2.1800 | 0.1667 | 0.2417 |
| upload_freq_90d | 1.0430 | 0.1000 | 0.4111 |
| freq_trend_ratio | 1.7191 | 1.6667 | 0.5838 |
| momentum_ratio | 3.6705 | 7.0272 | 3.5004 |
| avg_engagement_rate | 0.0232 | 0.0416 | 0.0789 |
| days_since_last_upload | 0.3333 | 2.0000 | 27.5000 |
| upload_regularity | 1.2099 | 25.5902 | 5.5582 |

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
