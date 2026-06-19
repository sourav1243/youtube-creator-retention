# Decisions

## Phase 1 — Environment & Scaffolding

### Decision: Project structure
- **What:** Create the full directory tree as specified in the build guide before writing any business logic.
- **Why:** Ensures all future phases have a pre-defined home for their artifacts, preventing scattered files.
- **Alternative:** Letting the structure emerge organically — risk of inconsistency.

### Decision: Config loading
- **What:** Use `python-dotenv` for secrets (loaded from `.env`) and `PyYAML` for typed config from `config/config.yaml`.
- **Why:** Separates secrets from configuration; follows 12-factor app principles.
- **Alternative:** Single `.env` file for everything — conflates secrets with settings.

### Decision: Virtual environment management
- **What:** Use Python venv with pinned `requirements.txt` produced by `pip freeze`.
- **Why:** Reproducible builds; exact versions avoid "works on my machine" issues.
- **Alternative:** Poetry/Pipenv — adds complexity for a learning/pipeline project.

### Decision: Tooling choice
- **What:** `ruff` for linting, `pytest` + `pytest-cov` for testing.
- **Why:** `ruff` is orders of magnitude faster than flake8; `pytest` is the standard for Python.
- **Alternative:** `black` + `flake8` — slower linting pipeline with equivalent results.

---

## Phase 3 — Extraction Pipeline

### Decision: Tenacity retry strategy
- **What:** Exponential backoff (2s base, 60s max) on HTTP 403/429/5xx. `QuotaExhausted` exception raised cleanly on confirmed `quotaExceeded`/`dailyLimitExceeded` — not retried.
- **Why:** Retrying a dead quota wastes calls and time. The manifest checkpoint allows resumption on the next day.
- **Alternative:** Retrying everything — burns remaining quota on pointless retries.

### Decision: Manifest format
- **What:** Simple CSV with columns: channel_id, stage, status, fetched_at. Updated after every successful batch.
- **Why:** Human-readable, diffable, easy to debug. No database dependency needed for the extraction phase.
- **Alternative:** JSON manifest — less human-readable; SQLite — overkill for a checkpoint file.

### Decision: playlistItems pagination default
- **What:** No page limit by default (max_pages=None) — pages through every page. Added max_pages parameter for testing.
- **Why:** YouTube channels can have thousands of videos; truncating would miss data. Testing can limit pages.
- **Alternative:** Hard limit of 500 items — would miss older videos for large channels.

### Decision: Seed channel source
- **What:** Curated list of 40 real channel IDs across 6 niches (tech, gaming, education, music, finance, entertainment/news) with verified `UC[A-Za-z0-9_-]{22}` format. Sourced from public Hugging Face dataset (ytRankAI/Top_100_YouTube_Channels) and verified YouTube channel handles.
- **Why:** Avoids spending 100 units/call on `search.list` to discover channels. The live extraction pipeline is the actual value — borrowing public channel IDs as seeds is explicitly endorsed by the build guide.
- **Alternative:** Using `search.list` API — costs 100 units for only 50 noisy results; wasteful for seed discovery.

### Decision: N_CHANNELS and tiering strategy
- **What:** N = 5,000 total channels (config default). Tier A (channels.list) for all 5,000 = 100 units. Tier B (video-level) for a stratified random sample of 1,000 channels = 8,000 units (4,000 playlistItems + 4,000 videos.list). Total Tier B cost: 8,100 units, fitting comfortably within the 10,000 daily quota with 1,900 units of headroom for retries.
- **Why:** Extracting video-level data for all 5,000 channels (40,000+ units) would require 4+ days. A 1,000-channel sample is large enough for meaningful clustering while fitting in one day.
- **Alternative:** Full extraction over multiple days using the resumable manifest — feasible but adds complexity for the first pass. Documented as future work.

### Decision: Estimated avg videos/channel
- **What:** 200 videos/channel default assumption, to be updated after measuring actual data from the sample run.
- **Why:** A reasonable estimate for established channels. Will be replaced with actual measured average after extraction.
- **Alternative:** Using a lower/higher estimate — either could under/over-provision quota headroom.

### Decision: Compliance note
- **What:** Added note in README about YouTube API Services Terms of Service for data caching/retention and refresh requirements.
- **Why:** General compliance hygiene for a learning project that touches real API data.
