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

    if risk == "Unscored":
        return "Insufficient data — collect more upload history"

    days = float(days) if pd.notna(days) else 0.0
    momentum = float(momentum) if pd.notna(momentum) else 1.0

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


def _check_data_freshness(clusters_df: pd.DataFrame, features_df: pd.DataFrame) -> bool:
    if clusters_df.empty or features_df.empty:
        logger.error("Clusters or features DataFrame is empty — cannot generate report")
        return False
    cluster_channels = set(clusters_df["channel_id"])
    feature_channels = set(features_df["channel_id"])
    overlap = cluster_channels & feature_channels
    if len(overlap) < max(len(cluster_channels), len(feature_channels)):
        logger.warning(
            "Data staleness detected: clusters has %d channels, features has %d, overlap: %d",
            len(cluster_channels),
            len(feature_channels),
            len(overlap),
        )
    return len(overlap) > 0


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

    if not _check_data_freshness(clusters_df, features_df):
        logger.error("Cannot generate report — data mismatch between clusters and features")
        return pd.DataFrame()

    try:
        channels_df = pd.read_parquet(channels_parquet)[["channel_id", "title"]]
    except (FileNotFoundError, OSError):
        channels_df = None

    clusters_for_merge = clusters_df.drop(
        columns=[c for c in features_df.columns if c in clusters_df.columns and c != "channel_id"],
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

    float_cols = [
        "upload_freq_30d",
        "upload_freq_90d",
        "momentum_ratio",
        "avg_engagement_rate",
        "upload_regularity",
        "duration_trend",
        "risk_score",
    ]
    for col in float_cols:
        if col in merged.columns:
            merged[col] = merged[col].round(4)

    output_cols = [
        "channel_id",
        "title",
        "risk_flag",
        "risk_score",
        "cluster_label",
        "upload_freq_30d",
        "upload_freq_90d",
        "momentum_ratio",
        "avg_engagement_rate",
        "days_since_last_upload",
        "upload_regularity",
        "duration_trend",
        "distance_to_centroid",
        "recommended_action",
    ]

    report = merged[output_cols].copy()
    report = report.sort_values(
        by=["risk_flag", "risk_score", "days_since_last_upload"],
        ascending=[True, False, False],
    )

    output_csv = Path(output_csv or ROOT_DIR / "reports" / "at_risk_creators.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False, float_format="%.4f", na_rep="")
    logger.info("Report saved to %s (%d rows)", output_csv, len(report))

    return report


def run_reporting_pipeline() -> pd.DataFrame:
    report = generate_report()
    if report.empty:
        return report
    risk_counts = report["risk_flag"].value_counts()
    logger.info("Risk distribution:\n%s", risk_counts)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_reporting_pipeline()
    print(report.head(10).to_string())
