from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)


def _recommended_action(row: dict) -> str:
    days = row.get("days_since_last_upload")
    risk = str(row.get("risk_flag", ""))
    momentum = row.get("momentum_ratio")

    if not days or pd.isna(days):
        days = 0
    if not momentum or pd.isna(momentum):
        momentum = 1.0
    days = float(days)
    momentum = float(momentum)

    if risk == "At-Risk":
        if days > 21:
            return "Reach out — momentum declining, no upload in 21+ days"
        elif momentum < 0.5:
            return "Reach out — momentum sharply declining"
        else:
            return "Monitor — declining engagement trend"
    elif risk == "Watch":
        if days > 14:
            return "Check in — upload gap widening"
        else:
            return "Monitor — mid-range metrics"
    else:
        return "No action needed"


def generate_report(
    clusters_parquet: str | Path | None = None,
    features_parquet: str | Path | None = None,
    output_csv: str | Path | None = None,
    channels_parquet: str | Path | None = None,
) -> pd.DataFrame:
    clusters_parquet = Path(clusters_parquet or ROOT_DIR / "data" / "processed" / "creator_clusters.parquet")
    features_parquet = Path(features_parquet or ROOT_DIR / "data" / "processed" / "creator_features.parquet")
    channels_parquet = Path(channels_parquet or ROOT_DIR / "data" / "processed" / "channels_clean.parquet")

    clusters_df = pd.read_parquet(clusters_parquet)
    features_df = pd.read_parquet(features_parquet)

    # Load channel titles for human-readable report (graceful fallback)
    try:
        channels_df = pd.read_parquet(channels_parquet)[["channel_id", "title"]]
    except (FileNotFoundError, OSError):
        channels_df = None

    # Drop feature columns from clusters_df so merge brings them cleanly from features_df
    feature_cols_in_clusters = [
        "upload_freq_30d", "upload_freq_90d", "freq_trend_ratio",
        "momentum_ratio", "avg_engagement_rate", "days_since_last_upload",
        "insufficient_history", "computed_at",
    ]
    clusters_for_merge = clusters_df.drop(
        columns=[c for c in feature_cols_in_clusters if c in clusters_df.columns],
        errors="ignore",
    )

    merged = clusters_for_merge.merge(
        features_df,
        on="channel_id",
        how="left",
    )
    if channels_df is not None:
        merged = merged.merge(channels_df, on="channel_id", how="left")

    merged["recommended_action"] = merged.apply(_recommended_action, axis=1)

    output_cols = [
        "channel_id",
        "title",
        "risk_flag",
        "cluster_label",
        "upload_freq_30d",
        "upload_freq_90d",
        "momentum_ratio",
        "avg_engagement_rate",
        "days_since_last_upload",
        "distance_to_centroid",
        "recommended_action",
    ]

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
