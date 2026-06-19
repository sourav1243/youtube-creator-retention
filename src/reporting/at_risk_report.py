from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)


def _recommended_action(row: dict) -> str:
    days = row.get("days_since_last_upload", 0)
    risk = row.get("risk_flag", "")
    momentum = row.get("momentum_ratio", 1.0)

    if risk == "At-Risk":
        if days is not None and days > 21:
            return "Reach out — momentum declining, no upload in 21+ days"
        elif momentum is not None and momentum < 0.5:
            return "Reach out — momentum sharply declining"
        else:
            return "Monitor — declining engagement trend"
    elif risk == "Watch":
        if days is not None and days > 14:
            return "Check in — upload gap widening"
        else:
            return "Monitor — mid-range metrics"
    else:
        return "No action needed"


def generate_report(
    clusters_parquet: str | Path | None = None,
    features_parquet: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    clusters_parquet = Path(clusters_parquet or ROOT_DIR / "data" / "processed" / "creator_clusters.parquet")
    features_parquet = Path(features_parquet or ROOT_DIR / "data" / "processed" / "creator_features.parquet")

    clusters_df = pd.read_parquet(clusters_parquet)
    features_df = pd.read_parquet(features_parquet)

    merged = clusters_df.merge(
        features_df,
        on="channel_id",
        how="left",
        suffixes=("_cluster", "_feature"),
    )

    merged["recommended_action"] = merged.apply(_recommended_action, axis=1)

    output_cols = [
        "channel_id",
        "risk_flag",
        "cluster_label",
        "upload_freq_30d",
        "upload_freq_90d",
        "momentum_ratio",
        "avg_engagement_rate",
        "days_since_last_upload",
        "recommended_action",
    ]

    for col in output_cols:
        if col not in merged.columns:
            merged[col] = None

    report = merged[output_cols].copy()
    report = report.sort_values(
        by=["risk_flag", "days_since_last_upload"],
        ascending=[True, False],
    )

    output_csv = Path(output_csv or ROOT_DIR / "reports" / "at_risk_creators.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    logger.info("Report saved to %s (%d rows)", output_csv, len(report))

    return report


def run_reporting_pipeline() -> pd.DataFrame:
    report = generate_report()
    risk_counts = report["risk_flag"].value_counts()
    logger.info("Risk distribution:\n%s", risk_counts)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_reporting_pipeline()
    print(report.head(10).to_string())
