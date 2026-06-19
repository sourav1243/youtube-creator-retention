# YouTube Creator Retention & Clustering

An end-to-end data pipeline that extracts YouTube channel and video metrics via the YouTube Data API v3, engineers momentum/engagement features, and applies unsupervised K-Means clustering to segment creators into actionable risk tiers.

## Problem

Partner Managers need to know *before* a creator goes dark which channels are losing momentum, so they can intervene proactively rather than discovering churn after the fact.

## Approach

Pull live per-channel and per-video metrics via the YouTube Data API v3, engineer momentum/engagement features from raw JSON, and use unsupervised K-Means clustering to segment creators into actionable risk tiers — without needing labeled "churned/not churned" data.

## Architecture

```
YouTube Data API v3 → Raw JSON on disk → MySQL (system of record)
                                                        ↓
                                              DuckDB analytical layer
                                                        ↓
                                         Feature Engineering (Pandas)
                                                        ↓
                                          K-Means Clustering (sklearn)
                                                        ↓
                                     at_risk_creators.csv → Power BI
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Access | `requests` + `tenacity` |
| Cleaning / Feature Engineering | Pandas, NumPy |
| System of Record | MySQL |
| Analytical Layer | DuckDB |
| Modeling | Scikit-learn (KMeans, RobustScaler) |
| Testing | pytest, pytest-cov |
| CI | GitHub Actions |
| BI | Power BI |

## Setup

```bash
# Clone and enter the repo
cd youtube-creator-retention

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your YouTube API key and MySQL credentials

# Run the full pipeline
python -m src.run_pipeline
```

## Phases

1. **Environment & Scaffolding** — Project skeleton, config, virtualenv
2. **API Access & Seed Channels** — Channel ID list, quota budget
3. **Extraction Pipeline** — Resumable, quota-aware YouTube API client
4. **MySQL Schema & Load** — System of record with upsert semantics
5. **Cleaning & Feature Engineering** — 30/90-day rolling features, momentum proxy
6. **DuckDB Analytical Layer** — Fast local analytical workspace
7. **K-Means Clustering** — Business-labeled risk tiers
8. **At-Risk Reporting** — Partner-Manager-facing CSV output
9. **Power BI Dashboard** — Data model + build guide
10. **Testing, CI & Documentation** — Coverage, linting, reproducibility

## Resume Bullet Traceability

| Resume Bullet | Proven By |
|---|---|
| "Engineered a data extraction pipeline using the YouTube Data API v3 and Python to pull and parse historical metrics for 5,000+ creator channels." | Phases 2–4: `seed_channels.py`, `quota_planner.py`, `youtube_client.py`, `extract_channels.py`/`extract_videos.py`, MySQL schema + load with reconciliation |
| "Cleaned raw JSON payloads with Pandas, handling null timestamps and erratic view counts to calculate 30-day rolling upload frequencies." | Phase 5: `clean.py`, `engineer.py`, unit tests with hand-calculated fixtures, `reports/eda_summary.md` |
| "Applied K-Means clustering via Scikit-learn to segment creators based on engagement momentum, enabling Partner Managers to proactively identify and support 'At-Risk' channels." | Phases 7–9: `cluster.py`, `model_card.md`, `reports/at_risk_creators.csv`, Power BI `BUILD_GUIDE.md` |

## Compliance

Review YouTube's [API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service) for data caching/retention and refresh requirements before using this beyond a learning project. Do not publicly redistribute raw pulled payloads.
