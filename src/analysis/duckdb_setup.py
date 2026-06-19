from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)


def setup_duckdb(
    parquet_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> duckdb.DuckDBPyConnection:
    parquet_path = Path(parquet_path or ROOT_DIR / "data" / "processed" / "creator_features.parquet")
    db_path = Path(db_path or ROOT_DIR / "data" / "processed" / "creator_analytics.duckdb")

    conn = duckdb.connect(str(db_path))

    conn.execute("INSTALL parquet; LOAD parquet;")
    conn.execute(f"CREATE OR REPLACE VIEW creator_features AS SELECT * FROM read_parquet('{parquet_path}')")

    logger.info("DuckDB database created at %s", db_path)
    return conn


def try_attach_mysql(conn: duckdb.DuckDBPyConnection) -> bool:
    try:
        conn.execute("INSTALL mysql; LOAD mysql;")
        from src.config import settings

        dsn = settings.mysql.dsn
        conn.execute(f"ATTACH '{dsn}' AS mysql_db (TYPE mysql)")
        logger.info("MySQL attached successfully via DuckDB mysql extension")
        return True
    except Exception as e:
        logger.warning("MySQL attach failed (non-fatal): %s", e)
        logger.info("Falling back to Parquet-only mode")
        return False


def run_eda(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    results: dict[str, Any] = {}

    results["row_count"] = conn.execute("SELECT COUNT(*) FROM creator_features").fetchone()[0]
    results["insufficient_count"] = conn.execute(
        "SELECT COUNT(*) FROM creator_features WHERE insufficient_history = TRUE"
    ).fetchone()[0]

    results["feature_summary"] = conn.execute("""
        SELECT
            'upload_freq_30d' AS feature,
            COUNT(*) AS n,
            ROUND(AVG(upload_freq_30d), 4) AS mean,
            ROUND(STDDEV(upload_freq_30d), 4) AS std,
            ROUND(MIN(upload_freq_30d), 4) AS min,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY upload_freq_30d), 4) AS p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY upload_freq_30d), 4) AS median,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY upload_freq_30d), 4) AS p75,
            ROUND(MAX(upload_freq_30d), 4) AS max
        FROM creator_features WHERE upload_freq_30d IS NOT NULL
        UNION ALL
        SELECT
            'upload_freq_90d',
            COUNT(*),
            ROUND(AVG(upload_freq_90d), 4),
            ROUND(STDDEV(upload_freq_90d), 4),
            ROUND(MIN(upload_freq_90d), 4),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY upload_freq_90d), 4),
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY upload_freq_90d), 4),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY upload_freq_90d), 4),
            ROUND(MAX(upload_freq_90d), 4)
        FROM creator_features WHERE upload_freq_90d IS NOT NULL
        UNION ALL
        SELECT
            'freq_trend_ratio',
            COUNT(*),
            ROUND(AVG(freq_trend_ratio), 4),
            ROUND(STDDEV(freq_trend_ratio), 4),
            ROUND(MIN(freq_trend_ratio), 4),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY freq_trend_ratio), 4),
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY freq_trend_ratio), 4),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY freq_trend_ratio), 4),
            ROUND(MAX(freq_trend_ratio), 4)
        FROM creator_features WHERE freq_trend_ratio IS NOT NULL
        UNION ALL
        SELECT
            'momentum_ratio',
            COUNT(*),
            ROUND(AVG(momentum_ratio), 4),
            ROUND(STDDEV(momentum_ratio), 4),
            ROUND(MIN(momentum_ratio), 4),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY momentum_ratio), 4),
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY momentum_ratio), 4),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY momentum_ratio), 4),
            ROUND(MAX(momentum_ratio), 4)
        FROM creator_features WHERE momentum_ratio IS NOT NULL
        UNION ALL
        SELECT
            'avg_engagement_rate',
            COUNT(*),
            ROUND(AVG(avg_engagement_rate), 4),
            ROUND(STDDEV(avg_engagement_rate), 4),
            ROUND(MIN(avg_engagement_rate), 4),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY avg_engagement_rate), 4),
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY avg_engagement_rate), 4),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY avg_engagement_rate), 4),
            ROUND(MAX(avg_engagement_rate), 4)
        FROM creator_features WHERE avg_engagement_rate IS NOT NULL
        UNION ALL
        SELECT
            'days_since_last_upload',
            COUNT(*),
            ROUND(AVG(days_since_last_upload), 4),
            ROUND(STDDEV(days_since_last_upload), 4),
            ROUND(MIN(days_since_last_upload), 4),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY days_since_last_upload), 4),
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY days_since_last_upload), 4),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY days_since_last_upload), 4),
            ROUND(MAX(days_since_last_upload), 4)
        FROM creator_features WHERE days_since_last_upload IS NOT NULL
    """).fetchall()

    results["correlation"] = conn.execute("""
        SELECT
            ROUND(CORR(upload_freq_30d, momentum_ratio), 4) AS freq_momentum_corr,
            ROUND(CORR(upload_freq_30d, avg_engagement_rate), 4) AS freq_engagement_corr,
            ROUND(CORR(momentum_ratio, avg_engagement_rate), 4) AS momentum_engagement_corr,
            ROUND(CORR(days_since_last_upload, momentum_ratio), 4) AS days_momentum_corr,
            ROUND(CORR(days_since_last_upload, upload_freq_30d), 4) AS days_freq_corr
        FROM creator_features
        WHERE upload_freq_30d IS NOT NULL
          AND momentum_ratio IS NOT NULL
          AND avg_engagement_rate IS NOT NULL
    """).fetchone()

    return results


def generate_eda_report(results: dict[str, Any], output_path: str | Path | None = None):
    output_path = Path(output_path or ROOT_DIR / "reports" / "eda_summary.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EDA Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overview",
        "",
        f"- Total channels in feature table: {results['row_count']}",
        f"- Channels with insufficient history (unscored): {results['insufficient_count']}",
        f"- Scored channels: {results['row_count'] - results['insufficient_count']}",
        "",
        "## Feature Distributions",
        "",
        "| Feature | N | Mean | Std | Min | P25 | Median | P75 | Max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for row in results["feature_summary"]:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |"
        )

    if results["correlation"]:
        lines.extend([
            "",
            "## Feature Correlations",
            "",
            f"- Upload frequency (30d) vs Momentum ratio: {results['correlation'][0]}",
            f"- Upload frequency (30d) vs Engagement rate: {results['correlation'][1]}",
            f"- Momentum ratio vs Engagement rate: {results['correlation'][2]}",
            f"- Days since last upload vs Momentum ratio: {results['correlation'][3]}",
            f"- Days since last upload vs Upload frequency: {results['correlation'][4]}",
        ])

    lines.extend([
        "",
        "## Notes",
        "",
        "- Momentum ratio is computed as views-per-day proxy (see DECISIONS.md).",
        "- Channels with <2 videos in relevant windows flagged as insufficient_history.",
        "- This report is auto-generated by duckdb_setup.py — do not hand-edit.",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("EDA report generated: %s", output_path)


def setup_and_analyze(
    parquet_path: str | Path | None = None,
    db_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> duckdb.DuckDBPyConnection:
    conn = setup_duckdb(parquet_path, db_path)
    try_attach_mysql(conn)

    logger.info("Running EDA...")
    results = run_eda(conn)
    generate_eda_report(results, report_path)

    for row in results["feature_summary"]:
        logger.info(
            "  %s: mean=%.4f, median=%.4f, p25=%.4f, p75=%.4f",
            row[0], row[2], row[6], row[5], row[7],
        )

    if results["correlation"]:
        logger.info("  Correlations: freq-momentum=%.4f, momentum-engagement=%.4f", results["correlation"][0], results["correlation"][2])

    return conn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = setup_and_analyze()
    conn.close()
