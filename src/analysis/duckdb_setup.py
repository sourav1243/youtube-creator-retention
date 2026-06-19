from __future__ import annotations

import logging
from datetime import UTC, datetime
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

    if not parquet_path.exists():
        logger.error("Parquet file not found: %s — run feature engineering first", parquet_path)
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

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


def run_eda(conn: duckdb.DuckDBPyConnection, features_df: pd.DataFrame | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {}

    results["row_count"] = conn.execute("SELECT COUNT(*) FROM creator_features").fetchone()[0]
    results["insufficient_count"] = conn.execute(
        "SELECT COUNT(*) FROM creator_features WHERE insufficient_history = TRUE"
    ).fetchone()[0]

    # Data quality metrics from the features DataFrame directly
    if features_df is not None:
        scored = features_df[~features_df["insufficient_history"]]
        results["feature_nulls"] = {
            col: int(scored[col].isna().sum())
            for col in ["upload_freq_30d", "upload_freq_90d", "momentum_ratio", "avg_engagement_rate", "days_since_last_upload", "upload_regularity", "duration_trend"]
        }
        results["feature_zeros"] = {
            col: int((scored[col] == 0).sum())
            for col in ["upload_freq_30d", "upload_freq_90d", "days_since_last_upload", "upload_regularity", "duration_trend"]
        }
    else:
        results["feature_nulls"] = {}
        results["feature_zeros"] = {}

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
        UNION ALL
        SELECT
            'upload_regularity',
            COUNT(*),
            ROUND(AVG(upload_regularity), 4),
            ROUND(STDDEV(upload_regularity), 4),
            ROUND(MIN(upload_regularity), 4),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY upload_regularity), 4),
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY upload_regularity), 4),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY upload_regularity), 4),
            ROUND(MAX(upload_regularity), 4)
        FROM creator_features WHERE upload_regularity IS NOT NULL
        UNION ALL
        SELECT
            'duration_trend',
            COUNT(*),
            ROUND(AVG(duration_trend), 4),
            ROUND(STDDEV(duration_trend), 4),
            ROUND(MIN(duration_trend), 4),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY duration_trend), 4),
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_trend), 4),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY duration_trend), 4),
            ROUND(MAX(duration_trend), 4)
        FROM creator_features WHERE duration_trend IS NOT NULL
    """).fetchall()

    results["correlation"] = conn.execute("""
        SELECT
            ROUND(CORR(upload_freq_30d, momentum_ratio), 4) AS freq_momentum_corr,
            ROUND(CORR(upload_freq_30d, avg_engagement_rate), 4) AS freq_engagement_corr,
            ROUND(CORR(momentum_ratio, avg_engagement_rate), 4) AS momentum_engagement_corr,
            ROUND(CORR(days_since_last_upload, momentum_ratio), 4) AS days_momentum_corr,
            ROUND(CORR(days_since_last_upload, upload_freq_30d), 4) AS days_freq_corr,
            ROUND(CORR(upload_regularity, upload_freq_30d), 4) AS regularity_freq_corr,
            ROUND(CORR(duration_trend, momentum_ratio), 4) AS duration_momentum_corr
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
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Overview",
        "",
        f"- Total channels in feature table: {results['row_count']}",
        f"- Channels with insufficient history (unscored): {results['insufficient_count']}",
        f"- Scored channels: {results['row_count'] - results['insufficient_count']}",
        "",
        "## Data Quality",
        "",
    ]

    if results.get("feature_nulls"):
        lines.append("### Feature Null Counts (scored channels)")
        nulls = results["feature_nulls"]
        for col, count in nulls.items():
            pct = count / max(results["row_count"] - results["insufficient_count"], 1) * 100
            lines.append(f"- {col}: {count} nulls ({pct:.1f}%)")
        lines.append("")

    if results.get("feature_zeros"):
        lines.append("### Feature Zero Counts (scored channels)")
        zeros = results["feature_zeros"]
        for col, count in zeros.items():
            pct = count / max(results["row_count"] - results["insufficient_count"], 1) * 100
            lines.append(f"- {col}: {count} zeros ({pct:.1f}%)")
        lines.append("")

    lines.extend([
        "## Feature Distributions",
        "",
        "| Feature | N | Mean | Std | Min | P25 | Median | P75 | Max |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for row in results["feature_summary"]:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |"
        )

    if results["correlation"]:
        corr = results["correlation"]
        lines.extend([
            "",
            "## Feature Correlations",
            "",
        ])
        corr_labels = [
            ("Upload frequency (30d) vs Momentum ratio", corr[0]),
            ("Upload frequency (30d) vs Engagement rate", corr[1]),
            ("Momentum ratio vs Engagement rate", corr[2]),
            ("Days since last upload vs Momentum ratio", corr[3]),
            ("Days since last upload vs Upload frequency", corr[4]),
            ("Upload regularity vs Upload frequency (30d)", corr[5]),
            ("Duration trend vs Momentum ratio", corr[6]),
        ]
        for label, val in corr_labels:
            val_str = f"{val:.4f}" if val is not None else "N/A (insufficient data)"
            lines.append(f"- {label}: {val_str}")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Momentum ratio is computed as views-per-day proxy (see DECISIONS.md).",
        "- Channels with <2 videos in relevant windows flagged as insufficient_history.",
        "- This report is auto-generated by duckdb_setup.py — do not hand-edit.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("EDA report generated: %s", output_path)


def setup_and_analyze(
    parquet_path: str | Path | None = None,
    db_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> duckdb.DuckDBPyConnection:
    conn = setup_duckdb(parquet_path, db_path)
    try_attach_mysql(conn)

    # Load features DataFrame for data quality metrics
    import pandas as pd
    parquet_path_resolved = Path(parquet_path or ROOT_DIR / "data" / "processed" / "creator_features.parquet")
    features_df = pd.read_parquet(parquet_path_resolved) if parquet_path_resolved.exists() else None

    logger.info("Running EDA...")
    results = run_eda(conn, features_df=features_df)
    generate_eda_report(results, report_path)

    for row in results["feature_summary"]:
        logger.info(
            "  %s: mean=%.4f, median=%.4f, p25=%.4f, p75=%.4f",
            row[0], row[2], row[6], row[5], row[7],
        )

    if results["correlation"]:
        logger.info(
            "  Correlations: freq-momentum=%.4f, freq-engagement=%.4f, momentum-engagement=%.4f, days-momentum=%.4f, days-freq=%.4f, regularity-freq=%.4f, duration-momentum=%.4f",
            results["correlation"][0], results["correlation"][1], results["correlation"][2], results["correlation"][3], results["correlation"][4], results["correlation"][5], results["correlation"][6],
        )

    return conn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = setup_and_analyze()
    conn.close()
