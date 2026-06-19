from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.config import ROOT_DIR, settings
from src.features.clean import load_and_clean_channels, load_and_clean_videos

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator is None or numerator is None:
        return None
    if denominator <= 0:
        return None
    result = numerator / denominator
    if np.isinf(result) or np.isnan(result):
        return None
    return result


def _avg_views_per_day(
    video_subset: pd.DataFrame,
    reference_date: datetime,
) -> float | None:
    if len(video_subset) == 0:
        return None
    days_since = (reference_date - video_subset["published_at"]).dt.days.clip(lower=1)
    vpd = video_subset["view_count"] / days_since
    vpd = vpd.replace([np.inf, -np.inf], np.nan)
    if vpd.isna().all():
        return None
    return float(vpd.mean())


def _engagement_rate(video_subset: pd.DataFrame) -> float | None:
    if len(video_subset) == 0:
        return None
    likes = video_subset["like_count"].fillna(0)
    comments = video_subset["comment_count"].fillna(0)
    likes_comments = likes + comments
    views = video_subset["view_count"].replace(0, np.nan)
    if views.isna().all():
        return None
    rates = (likes_comments / views).clip(0, 1)
    return float(rates.mean())


def compute_features(
    channels_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    if reference_date is None:
        reference_date = datetime.now(UTC)

    if channels_df.empty:
        logger.warning("No channel data provided to feature engineer")
        return pd.DataFrame()

    if videos_df.empty:
        logger.warning("No video data provided to feature engineer")
        return pd.DataFrame()

    w30 = settings.features.window_30d
    w90 = settings.features.window_90d
    min_videos = settings.features.min_videos_for_scoring

    cutoff_30d = reference_date - pd.Timedelta(days=w30)
    cutoff_90d = reference_date - pd.Timedelta(days=w90)

    # Pre-compute window indicators for vectorized groupby
    videos_df = videos_df.copy()
    videos_df["in_30d"] = videos_df["published_at"] >= cutoff_30d
    videos_df["in_90d"] = videos_df["published_at"] >= cutoff_90d
    videos_df["in_31_90d"] = videos_df["in_90d"] & ~videos_df["in_30d"]

    # Group-level aggregations
    grouped = videos_df.groupby("channel_id")

    n_30d = grouped["in_30d"].sum()
    n_90d = grouped["in_90d"].sum()

    upload_freq_30d = n_30d / w30
    upload_freq_90d = n_90d / w90

    last_upload = grouped["published_at"].max()
    days_since = last_upload.apply(lambda x: max((reference_date - x).days, 0) if pd.notna(x) else None)

    features_rows: list[dict] = []

    for cid in channels_df["channel_id"].unique():
        total_videos = n_30d.get(cid, 0) + (n_90d.get(cid, 0) - n_30d.get(cid, 0))

        if total_videos == 0:
            features_rows.append(
                {
                    "channel_id": cid,
                    "computed_at": reference_date,
                    "upload_freq_30d": None,
                    "upload_freq_90d": None,
                    "momentum_ratio": None,
                    "avg_engagement_rate": None,
                    "days_since_last_upload": None,
                    "upload_regularity": None,
                    "duration_trend": None,
                    "insufficient_history": True,
                }
            )
            continue

        channel_videos = videos_df[videos_df["channel_id"] == cid]
        videos_last_30d = channel_videos[channel_videos["in_30d"]]
        videos_31_90d = channel_videos[channel_videos["in_31_90d"]]

        n30 = int(n_30d.get(cid, 0))
        n90 = int(n_90d.get(cid, 0))

        # Momentum ratio
        recent_vpd = _avg_views_per_day(videos_last_30d, reference_date)
        older_vpd = _avg_views_per_day(videos_31_90d, reference_date)

        if recent_vpd is not None and older_vpd is not None and older_vpd > 0:
            momentum_ratio = _safe_ratio(recent_vpd, older_vpd)
        elif len(channel_videos) >= 4:
            sorted_vids = channel_videos.sort_values("published_at")
            mid = len(sorted_vids) // 2
            older_half = sorted_vids.iloc[:mid]
            recent_half = sorted_vids.iloc[mid:]
            older_vpd_fb = _avg_views_per_day(older_half, reference_date)
            recent_vpd_fb = _avg_views_per_day(recent_half, reference_date)
            if recent_vpd_fb is not None and older_vpd_fb is not None and older_vpd_fb > 0:
                momentum_ratio = _safe_ratio(recent_vpd_fb, older_vpd_fb)
            else:
                momentum_ratio = None
        else:
            momentum_ratio = None

        # Engagement rate
        avg_engagement_rate = _engagement_rate(channel_videos)

        # Upload regularity: std of days between consecutive uploads
        if len(channel_videos) >= 3:
            sorted_uploads = channel_videos.sort_values("published_at")
            gaps = sorted_uploads["published_at"].diff().dt.days.iloc[1:]
            upload_regularity = float(gaps.std()) if len(gaps) > 0 else None
        else:
            upload_regularity = None

        # Duration trend: average video duration (minutes) for recent (30d) videos
        target_dur = videos_last_30d if len(videos_last_30d) > 0 else channel_videos
        duration_minutes = target_dur["duration_seconds"] / 60.0
        duration_trend = float(duration_minutes.mean()) if len(target_dur) > 0 else None

        # Insufficient history
        insufficient = bool(n30 < min_videos and n90 < min_videos)

        f30 = float(upload_freq_30d.get(cid, 0))
        f90 = float(upload_freq_90d.get(cid, 0))
        freq_trend_ratio = _safe_ratio(f30, f90)

        features_rows.append(
            {
                "channel_id": cid,
                "computed_at": reference_date,
                "upload_freq_30d": f30,
                "upload_freq_90d": f90,
                "freq_trend_ratio": freq_trend_ratio,
                "momentum_ratio": momentum_ratio,
                "avg_engagement_rate": avg_engagement_rate,
                "days_since_last_upload": days_since.get(cid),
                "upload_regularity": upload_regularity,
                "duration_trend": duration_trend,
                "insufficient_history": insufficient,
            }
        )

    result = pd.DataFrame(features_rows)
    logger.info(
        "Features computed: %d channels, %d with insufficient history",
        len(result),
        result["insufficient_history"].sum(),
    )
    return result


def save_histograms(features_df: pd.DataFrame, output_dir: str | Path | None = None):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir or ROOT_DIR / "reports" / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old histograms before saving new ones
    for old_file in output_dir.glob("*_histogram.png"):
        old_file.unlink(missing_ok=True)

    numeric_cols = [
        "upload_freq_30d",
        "upload_freq_90d",
        "freq_trend_ratio",
        "momentum_ratio",
        "avg_engagement_rate",
        "days_since_last_upload",
        "upload_regularity",
        "duration_trend",
    ]

    for col in numeric_cols:
        if col not in features_df.columns:
            continue
        series = features_df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(series, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.savefig(output_dir / f"{col}_histogram.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    logger.info("Histograms saved to %s", output_dir)


def save_features(
    features_df: pd.DataFrame,
    parquet_path: str | Path | None = None,
):
    parquet_path = Path(parquet_path or ROOT_DIR / "data" / "processed" / "creator_features.parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(parquet_path, index=False)
    logger.info("Features saved to %s", parquet_path)
    return parquet_path


def run_feature_pipeline(
    channels_dir: str | Path | None = None,
    videos_dir: str | Path | None = None,
) -> pd.DataFrame:
    channels_df = load_and_clean_channels(channels_dir)
    videos_df = load_and_clean_videos(videos_dir)

    processed_dir = ROOT_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    channels_df.to_parquet(processed_dir / "channels_clean.parquet", index=False)
    videos_df.to_parquet(processed_dir / "videos_clean.parquet", index=False)

    features_df = compute_features(channels_df, videos_df)
    save_features(features_df)

    return features_df
