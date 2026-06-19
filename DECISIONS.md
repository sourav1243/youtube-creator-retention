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

## Phase 2 — _to be decided_
