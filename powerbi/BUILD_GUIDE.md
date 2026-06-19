# Power BI Dashboard Build Guide

## Data Source Options

### Option A: CSV Import (simplest)

Two CSV files exported from the pipeline:

1. **`dim_channel.csv`** — Channel dimension
2. **`fact_creator_metrics.csv`** — Fact table with metrics, cluster assignments, and risk flags

Re-export from the pipeline:
```bash
python -c "
import pandas as pd
from src.config import ROOT_DIR

clusters = pd.read_parquet(ROOT_DIR / 'data' / 'processed' / 'creator_clusters.parquet')
features = pd.read_parquet(ROOT_DIR / 'data' / 'processed' / 'creator_features.parquet')

merged = clusters.merge(features, on='channel_id', how='left')

dim = merged[['channel_id', 'risk_flag', 'cluster_label', 'model_version']].drop_duplicates('channel_id')
dim.to_csv(ROOT_DIR / 'reports' / 'dim_channel.csv', index=False)

fact = merged[['channel_id', 'upload_freq_30d', 'upload_freq_90d', 'freq_trend_ratio',
                'momentum_ratio', 'avg_engagement_rate', 'days_since_last_upload']]
fact['scored_at'] = pd.Timestamp.utcnow()
fact.to_csv(ROOT_DIR / 'reports' / 'fact_creator_metrics.csv', index=False)

print('CSV files exported to reports/')
"
```

### Option B: MySQL Direct Connection (recommended for refresh)

Connect Power BI to MySQL using the MySQL connector:
- Server: `your_mysql_host`
- Database: `youtube_creator_retention`
- Tables: `channels`, `creator_features`, `creator_clusters`
- Authentication: Use the credentials from `.env`

## Relationship Model

```
fact_creator_metrics[channel_id] ---> dim_channel[channel_id] (many-to-one)
```

| Table | Type | Description |
|---|---|---|
| `dim_channel` | Dimension | Channel attributes, risk flags, cluster labels |
| `fact_creator_metrics` | Fact | Numerical metrics, engineering features |

## DAX Measures

```dax
-- Count of At-Risk creators
At-Risk Count =
CALCULATE(
    COUNTROWS(fact_creator_metrics),
    fact_creator_metrics[risk_flag] = "At-Risk"
)

-- Percentage of creators who are At-Risk
At-Risk % =
DIVIDE(
    [At-Risk Count],
    COUNTROWS(fact_creator_metrics)
)

-- Average momentum by cluster label
Avg Momentum by Cluster =
AVERAGEX(
    VALUES(fact_creator_metrics[cluster_label]),
    AVERAGE(fact_creator_metrics[momentum_ratio])
)

-- Average upload frequency (30d)
Avg Upload Freq 30d =
AVERAGE(fact_creator_metrics[upload_freq_30d])

-- Creators with no upload in 14+ days
Dormant Creators =
CALCULATE(
    COUNTROWS(fact_creator_metrics),
    fact_creator_metrics[days_since_last_upload] > 14
)

-- Risk breakdown
Risk Breakdown =
VALUES(dim_channel[risk_flag])
```

## Suggested Visual Layout

```
+----------------------------------------------------------+
|  KPI Row: [At-Risk Count] [At-Risk %] [Total Creators]   |
+----------------------------------------------------------+
|                                                          |
|  Scatter Plot:                                           |
|  X-axis: upload_freq_30d                                 |
|  Y-axis: momentum_ratio                                  |
|  Legend: cluster_label                                   |
|  Size: subscriber_count                                  |
|                                                          |
+----------------------------------------------------------+
|  Slicers: [Cluster] [Risk Flag] [Niche/Category]         |
+----------------------------------------------------------+
|  Table: At-Risk Creators (drill-through)                 |
|  Columns: Channel Title, Risk Flag,                      |
|           Days Since Last Upload, Recommended Action      |
+----------------------------------------------------------+
```

## Step-by-Step in Power BI Desktop

1. **Get Data** → CSV or MySQL
2. **Model view**: Create relationship between `dim_channel[channel_id]` and `fact_creator_metrics[channel_id]`
3. **New measures**: Copy the DAX measures above into the fact table
4. **Build visuals**:
   - 3 Card visuals for KPI row
   - Scatter chart with cluster labels as legend
   - Slicers for cluster and risk flag
   - Table visual for drill-through
5. **Publish** to Power BI Service for sharing with Partner Managers
