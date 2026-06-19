# Progress

## Phase 1 — Environment & Scaffolding
- [x] git init, .gitignore
- [x] Full directory tree created
- [x] Virtualenv created, dependencies installed
- [x] `src/config.py` loads .env + config.yaml
- [x] .env.example created
- [x] README.md, DECISIONS.md, PROGRESS.md initialized
- [x] `python -m src.config` runs without errors
- [x] .env is in .gitignore and was never committed
- [x] Directory tree matches Section 4
- [x] requirements.txt is pinned

## Phase 2 — API Access, Seed Channels & Quota Budgeting
- [x] `seed_channel_ids.csv` exists with 40 validated, deduplicated IDs across 6 niches
- [x] `quota_planner.py` outputs computed Tier A/Tier B plan
- [x] Tiering decision and numbers documented in DECISIONS.md
- [x] Compliance note added to README about YouTube API ToS
- [x] API key configured in .env

## Phase 3 — Extraction Pipeline
- [x] `youtube_client.py`: thin requests wrapper with tenacity retry, batch up to 50 IDs
- [x] `extract_channels.py`: Tier A channel extraction with manifest checkpoint
- [x] `extract_videos.py`: Tier B video extraction per channel
- [x] Manifest CSV for resumability across quota-interrupted runs
- [x] Verified on 5-channel sample: channels, playlist items, and video details all return correctly
- [x] 403 quotaExceeded triggers clean QuotaExhausted exception (not infinite retry)
- [x] 7 pytest tests pass (seeds, client, batching, quota error handling)

## Phase 4 — MySQL Schema & Load
- [ ] _pending_

## Phase 5 — Cleaning & Feature Engineering
- [ ] _pending_

## Phase 6 — DuckDB Analytical Layer
- [ ] _pending_

## Phase 7 — K-Means Clustering
- [ ] _pending_

## Phase 8 — At-Risk Reporting Layer
- [ ] _pending_

## Phase 9 — Power BI Dashboard
- [ ] _pending_

## Phase 10 — Testing, CI & Documentation
- [ ] _pending_

## Phase 11 — Stretch: Real Historical Snapshots
- [ ] _pending_
