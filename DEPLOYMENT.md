# Deployment Guide

## Prerequisites

- Python 3.11+
- YouTube Data API v3 key (generate at https://console.cloud.google.com/apis/credentials)
- Docker (optional, for MySQL)
- 10,000 quota units/day (free tier)

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your YouTube API key and MySQL credentials

# 4. Generate seed channel list
python -m src.extraction.seed_channels

# 5. Run full pipeline
python -m src.run_pipeline
```

## Data Extraction Strategy

The pipeline uses a two-tier extraction strategy to work within YouTube API quota limits:

| Tier | API Call | Cost | Scope |
|------|----------|------|-------|
| A | channels.list | 1 unit/50 IDs | All channels |
| B | playlistItems.list + videos.list | 2 units/channel × pages | Up to 20 pages/channel |

With `max_pages_per_channel: 20` (config/config.yaml), each channel costs ~40 units. For 37 channels: ~1,480 units total (well within 10,000 daily limit).

To extract all channels in one day:
```bash
python -m src.run_pipeline
```

For larger channel sets (500+), the manifest checkpoint system allows multi-day extraction. Run the pipeline daily and it resumes where it left off.

## MySQL Setup (Optional)

```bash
# Start MySQL
docker-compose up -d

# Initialize schema
docker exec -i yt_retention_mysql mysql -u root -proot_password youtube_creator_retention < sql/schema.sql

# Run pipeline (loads data into MySQL automatically)
python -m src.run_pipeline
```

## Model Selection

The pipeline auto-selects the best clustering algorithm (KMeans, GMM, or DBSCAN) by silhouette score. To force a specific model:

```python
from src.modeling.cluster import run_clustering_pipeline
result = run_clustering_pipeline(model_type="kmeans")  # "kmeans", "gmm", "dbscan", or "auto"
```

## Scheduled Runs

### Linux (crontab)
```
0 9 * * 1 cd /path/to/project && /usr/bin/python3 -m src.run_pipeline
```

### Windows (Task Scheduler)
1. Open Task Scheduler
2. Create Basic Task → Weekly
3. Action: Start a Program → `C:\path\to\python.exe`
4. Arguments: `-m src.run_pipeline`
5. Start in: `C:\path\to\project`

## Monitoring

- **Drift detection**: `python -m infrastructure.drift_detection` (checks feature means and cluster distribution shifts)
- **Alerting**: Configure `SMTP_*` and `ALERT_EMAIL` env vars for email alerts on new At-Risk creators
- **Logs**: `logs/pipeline.log` (rotating, 10MB per file, 5 backups)

## Performance with Current Data (37 channels, 5,550 videos)

| Pipeline Stage | Time |
|----------------|------|
| Channel extraction | ~2s (cached) |
| Feature engineering | ~1s |
| DuckDB EDA | ~0.5s |
| Model selection (auto) | ~10s |
| Reporting | ~0.1s |
| **Total** | **~14s** |

## Known Limitations

- **37 channels** produces K=2-3 clusters with moderate silhouette (~0.52). Target 200+ channels for stable K=5
- All current videos are capped at 150/channel (old 3-page limit). Next full extraction with `max_pages_per_channel: 20` will get ~1,000/channel
- Bootstrap stability requires 500+ samples for reliable per-channel confidence intervals
- `engagement_quality` removed due to near-perfect correlation (r=0.999) with `avg_engagement_rate`
- MySQL requires `docker-compose up` before pipeline will populate `creator_features` and `creator_clusters` tables
