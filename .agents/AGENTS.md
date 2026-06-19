# Project-Scoped Rules for YouTube Creator Retention

This file contains style guidelines, behavioral constraints, and instructions for any AI agent working on this project. Always follow these rules.

## 1. Agent Operating Contract

1. **Plan before you code**: At the start of any new task or phase, update `PROGRESS.md` with a checkbox task list.
2. **Verify on small samples**: Before running extraction, cleaning, or SQL loads on the full dataset, test on a 5–10 channel sample first.
3. **No silent data loss**: Any transform that drops, nulls, or caps values must log *how many rows* and *why* to `logs/pipeline.log`.
4. **Idempotency**: All database loader operations (`src/load/load_mysql.py`) must use upsert syntax (`ON DUPLICATE KEY UPDATE` for MySQL, `ON CONFLICT` for SQLite) to prevent row duplication or corruption.
5. **Secrecy**: Keep API keys and credentials in `.env`. Never commit `.env` or data files.
6. **Tests**: Keep the `pytest` test suite green. Every module with logic (not just I/O glue) gets at least one test.

## 2. Database Dialect Fallback

- The project uses **MySQL** as the production system of record.
- If MySQL is not running or accessible, the loader `src/load/load_mysql.py` must fall back to a local **SQLite** database at `data/processed/youtube_creator_retention.db`.
- The database schema is initialized programmatically if SQLite is active.
- When editing database queries, always write dialect-conditional code checks:
  ```python
  if engine.dialect.name == "sqlite":
      # use SQLite ON CONFLICT syntax
  else:
      # use MySQL ON DUPLICATE KEY UPDATE syntax
  ```

## 3. Modeling & Evaluation Guidelines

- Maintain a fixed `random_state=42` across all models (K-Means, GMM, DBSCAN) for reproducibility.
- If a channel has fewer than 2 videos in the trailing window, set `insufficient_history = True` and route it to the "Unscored" cluster instead of imputing fake values.
- Clean all pandas `Timestamp` objects using `.to_pydatetime()` or convert to string before SQL execution to avoid SQLite parameter binding failures.
- Clean all float `NaN` or `NaT` values to `None` in the parameters dictionary passed to SQLAlchemy to prevent database-level binding interface errors.
- Evaluate model stability dynamically using the greedy label matching logic in `_audit_analysis.py` (which matches bootstrap clusters to original labels) rather than assuming labels are static.
