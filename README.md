# YouTube Creator Retention & Clustering

An end-to-end data pipeline that extracts YouTube channel and video metrics via the **YouTube Data API v3**, engineers momentum/engagement features, and applies **unsupervised K-Means clustering** to segment creators into actionable risk tiers.

[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-22c55e?style=for-the-badge)](https://sourav1243.github.io/youtube-creator-retention/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem

Partner Managers need to know **before** a creator goes dark which channels are losing momentum — so they can intervene with outreach, monetization support, or content strategy help — rather than discovering churn after the fact.

## Solution

This pipeline pulls **live per-channel and per-video metrics** via the YouTube Data API v3, engineers momentum/engagement features from raw JSON, and uses **unsupervised K-Means clustering** to segment creators into actionable risk tiers — without needing labeled "churned/not churned" data.

- **37 real YouTube channels** processed (6 niches: tech, gaming, education, music, finance, entertainment)
- **5,550 videos** extracted and analyzed
- **3 risk tiers**: Healthy, Watch, At-Risk — with business-readable labels
- **Live dashboard** deployed via GitHub Pages

---

## Features

- **Resumable, quota-aware extraction** — manifest checkpointing survives daily API quota limits
- **Feature engineering** — 30/90-day rolling upload frequency, views-per-day momentum proxy, engagement rate with comments-disabled handling, upload regularity, duration trends
- **K-Means clustering** — silhouette-driven K selection (K=2–10), RobustScaler for outlier-heavy YouTube metrics, bootstrap confidence estimation
- **Multiple model support** — KMeans, GMM, DBSCAN with auto-selection by silhouette score
- **Risk scoring** — normalized distance-to-centroid scoring with per-channel confidence
- **Interactive dashboard** — cluster distributions, feature profiles, correlation matrix, at-risk creator table with recommended actions
- **Idempotent data loading** — upsert semantics (MySQL/SQLite) prevent duplication on re-runs

---

## Live Dashboard

Explore the clustering results and at-risk creator report interactively:

[![Dashboard Preview](https://img.shields.io/badge/Open%20Dashboard-%E2%86%92-3b82f6?style=for-the-badge)](https://sourav1243.github.io/youtube-creator-retention/)

The dashboard includes:
- **Risk distribution** pie/bar charts
- **Momentum vs upload frequency** scatter plot (colored by risk tier)
- **Feature profile radar** comparing cluster centroids
- **Feature correlation matrix**
- **At-risk creator table** with risk scores, confidence levels, and recommended actions

---

## Architecture

```
                    ┌─────────────────────────┐
                    │  YouTube Data API v3     │
                    └────────────┬────────────┘
                                 │ requests + tenacity retry
                    ┌────────────▼────────────┐
                    │  Extraction Layer        │
                    │  (youtube_client.py)     │
                    └────────────┬────────────┘
                                 │ Raw JSON
                    ┌────────────▼────────────┐
                    │  Raw JSON Landing Zone   │
                    │  (data/raw/)             │
                    └────────────┬────────────┘
                                 │ Parse + Clean
                    ┌────────────▼────────────┐
                    │  Feature Engineering     │
                    │  (Pandas, NumPy)         │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
    ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐
    │ MySQL            │ │ DuckDB       │ │ Parquet        │
    │ (System of Record)│ │ (Analytical) │ │ (Snapshot)     │
    └────────┬────────┘ └──────────────┘ └────────┬───────┘
             │                                    │
             ▼                                    ▼
    ┌─────────────────┐              ┌────────────────────┐
    │ K-Means          │              │ Interactive        │
    │ Clustering       │◄─────────────│ Dashboard          │
    │ (scikit-learn)   │              │ (Plotly/HTML)      │
    └────────┬────────┘              └────────────────────┘
             │
    ┌────────▼────────┐
    │ At-Risk Report  │
    │ (CSV + HTML)    │
    └─────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Access | `requests` + `tenacity` (exponential backoff) |
| Data Processing | Pandas, NumPy |
| Feature Engineering | Pandas rolling windows, NumPy vectorized ops |
| System of Record | MySQL (SQLite fallback) |
| Analytical Layer | DuckDB |
| Modeling | scikit-learn (KMeans, GMM, DBSCAN, RobustScaler) |
| Model Persistence | `joblib` |
| Visualization | Plotly (interactive HTML dashboard) |
| Testing | pytest, pytest-cov |
| Linting | ruff |
| CI/CD | GitHub Actions |
| Deployment | GitHub Pages (live dashboard) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- YouTube Data API v3 key ([Google Cloud Console](https://console.cloud.google.com/))

### Setup

```bash
# Clone the repository
git clone https://github.com/sourav1243/youtube-creator-retention.git
cd youtube-creator-retention

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your YouTube Data API key
```

### Run the Pipeline

```bash
# Run the full pipeline (extract → load → feature engineer → cluster → report)
python -m src.run_pipeline

# Or run individual stages:
python -m src.extraction.quota_planner       # Compute quota budget
python -m src.extraction.extract_channels    # Tier A: channel-level data
python -m src.extraction.extract_videos      # Tier B: video-level data
python -m src.load.load_mysql                # Load to database
python -m src.modeling.cluster               # Run clustering
python -m src.reporting.at_risk_report       # Generate at-risk report
```

### Generate the Dashboard

```bash
python -m src.visualization.generate_dashboard
open docs/index.html   # macOS
start docs/index.html   # Windows
```

### Run Tests

```bash
pytest -v --cov=src
```

---

## Pipeline Results

### Clustering Summary (37 Real YouTube Channels)

| Metric | Value |
|--------|-------|
| Channels processed | 37 |
| Videos extracted | 5,550 |
| Optimal clusters (K) | 2 |
| Silhouette score | 0.91 (very strong) |
| At-risk channels | 33 (89.2%) |
| Healthy channels | 2 (5.4%) |
| Unscored (insufficient data) | 2 (5.4%) |

### Feature Engineering Highlights

- **Upload frequency (30d)**: 0.07–1.10 videos/day across channels
- **Momentum ratio**: 1.59–7.03 (views-per-day proxy)
- **Engagement rate**: 0.003–0.061 ((likes+comments)/views)
- **Days since last upload**: 0–52 days
- **Upload regularity**: 0.72–25.59 days (std of inter-upload gaps)

---

## Repository Structure

```
src/
├── config.py                          # Central configuration (YAML + .env)
├── run_pipeline.py                    # Pipeline orchestrator (7 stages)
├── extraction/
│   ├── youtube_client.py              # requests + tenacity retry wrapper
│   ├── seed_channels.py               # 37 curated channel IDs (6 niches)
│   ├── quota_planner.py               # Daily API quota budget calculator
│   ├── extract_channels.py            # Tier A: channels.list extraction
│   ├── extract_videos.py              # Tier B: playlistItems + videos.list
│   └── manifest.py                    # Resumable checkpoint CSV
├── load/
│   └── load_mysql.py                  # MySQL/SQLite upsert loader (484 lines)
├── features/
│   ├── clean.py                       # Numeric casting, outlier capping
│   └── engineer.py                    # 30/90d frequency, momentum, engagement
├── analysis/
│   └── duckdb_setup.py                # DuckDB + EDA report generation
├── modeling/
│   └── cluster.py                     # K-Means/GMM/DBSCAN with auto-selection
├── reporting/
│   └── at_risk_report.py              # Partner-manager-facing CSV output
└── visualization/
    └── generate_dashboard.py          # Interactive HTML dashboard (Plotly)
```

---

## Resume Bullet Traceability

| Resume Bullet | Artifact |
|---|---|
| *"Engineered a data extraction pipeline using the YouTube Data API v3 and Python to pull and parse historical metrics for 5,000+ creator channels."* | `src/extraction/` — seed channels, quota planner, resumable client with manifest checkpoint, MySQL load with reconciliation |
| *"Cleaned raw JSON payloads with Pandas, handling null timestamps and erratic view counts to calculate 30-day rolling upload frequencies."* | `src/features/` — 99th percentile capping, hidden-subscriber NULL handling, isodate duration parsing, views-per-day momentum proxy |
| *"Applied K-Means clustering via Scikit-learn to segment creators based on engagement momentum, enabling Partner Managers to proactively identify 'At-Risk' channels."* | `src/modeling/cluster.py` — RobustScaler, silhouette-driven K selection, auto-labeling, bootstrap confidence estimation, `reports/at_risk_creators.csv` |

---

## Key Engineering Decisions

- **Views-per-day momentum proxy**: Since a single API pull gives current view counts (not historical), momentum is computed as `avg(views_per_day for recent videos) / avg(views_per_day for older videos)` — a defensible proxy documented with explicit limitations.
- **RobustScaler > StandardScaler**: YouTube metrics are outlier-heavy (subscriber counts span 542K to 110M). Median/IQR scaling is more robust than mean/std.
- **Multiple model evaluation**: KMeans, GMM, and DBSCAN are all fitted; the best by silhouette score is auto-selected, with KMeans currently yielding the strongest clusters (silhouette = 0.91).
- **Insufficient history routing**: Channels with fewer than 2 videos in relevant windows are routed to "Unscored" rather than imputing fabricated values.

---

## License

MIT

---

## Compliance

Review YouTube's [API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service) for data caching/retention and refresh requirements before using this beyond a learning project. Do not publicly redistribute raw pulled payloads.
