from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)


def compute_features(
    channels_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

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

    features: list[dict] = []

    for cid in channels_df["channel_id"].unique():
        channel_videos = videos_df[videos_df["channel_id"] == cid].copy()
        total_videos = len(channel_videos)

        if total_videos == 0:
            features.append({
                "channel_id": cid,
                "computed_at": reference_date,
                "upload_freq_30d": None,
                "upload_freq_90d": None,
                "freq_trend_ratio": None,
                "momentum_ratio": None,
                "avg_engagement_rate": None,
                "days_since_last_upload": None,
                "insufficient_history": True,
            })
            continue

        videos_last_30d = channel_videos[channel_videos["published_at"] >= cutoff_30d]
        videos_last_90d = channel_videos[channel_videos["published_at"] >= cutoff_90d]
        videos_31_90d = channel_videos[
            (channel_videos["published_at"] >= cutoff_90d)
            & (channel_videos["published_at"] < cutoff_30d)
        ]

        n_30d = len(videos_last_30d)
        n_90d = len(videos_last_90d)

        # Upload frequencies (videos per day)
        upload_freq_30d = n_30d / w30
        upload_freq_90d = n_90d / w90

        # Frequency trend ratio
        if upload_freq_90d > 0:
            freq_trend_ratio = upload_freq_30d / upload_freq_90d
        else:
            freq_trend_ratio = None

        # Momentum ratio: views-per-day for recent vs older videos
        def _avg_views_per_day(video_subset: pd.DataFrame) -> float | None:
            if len(video_subset) == 0:
                return None
            days_since = (reference_date - video_subset["published_at"]).dt.days.replace(0, 1)
            vpd = video_subset["view_count"] / days_since
            return float(vpd.mean())

        recent_vpd = _avg_views_per_day(videos_last_30d)
        older_vpd = _avg_views_per_day(videos_31_90d)

        if recent_vpd is not None and older_vpd is not None and older_vpd > 0:
            momentum_ratio = recent_vpd / older_vpd
        elif len(channel_videos) >= 4:
            # Fallback: split available videos by median date when date-window comparison is impossible
            sorted_vids = channel_videos.sort_values("published_at")
            mid = len(sorted_vids) // 2
            older_half = sorted_vids.iloc[:mid]
            recent_half = sorted_vids.iloc[mid:]
            older_vpd_fb = _avg_views_per_day(older_half)
            recent_vpd_fb = _avg_views_per_day(recent_half)
            if recent_vpd_fb is not None and older_vpd_fb is not None and older_vpd_fb > 0:
                momentum_ratio = recent_vpd_fb / older_vpd_fb
            else:
                momentum_ratio = None
        else:
            momentum_ratio = None

        # Engagement rate
        def _engagement_rate(video_subset: pd.DataFrame) -> float | None:
            if len(video_subset) == 0:
                return None
            likes_comments = video_subset["like_count"] + video_subset["comment_count"]
            views = video_subset["view_count"].replace(0, np.nan)
            rates = (likes_comments / views).clip(0, 1)
            return float(rates.mean())

        avg_engagement_rate = _engagement_rate(channel_videos)

        # Days since last upload
        last_upload = channel_videos["published_at"].max()
        days_since_last = (reference_date - last_upload).days if pd.notna(last_upload) else None

        # Insufficient history
        insufficient = (n_30d < min_videos) and (n_90d < min_videos)

        features.append({
            "channel_id": cid,
            "computed_at": reference_date,
            "upload_freq_30d": upload_freq_30d,
            "upload_freq_90d": upload_freq_90d,
            "freq_trend_ratio": freq_trend_ratio,
            "momentum_ratio": momentum_ratio,
            "avg_engagement_rate": avg_engagement_rate,
            "days_since_last_upload": days_since_last,
            "insufficient_history": bool(insufficient),
        })

    result = pd.DataFrame(features)
    logger.info(
        "Features computed: %d channels, %d with insufficient history",
        len(result),
        result["insufficient_history"].sum(),
    )
    return result


def save_histograms(features_df: pd.DataFrame, output_dir: str | Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir or ROOT_DIR / "reports" / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = ["upload_freq_30d", "upload_freq_90d", "freq_trend_ratio", "momentum_ratio", "avg_engagement_rate", "days_since_last_upload"]

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
    from src.features.clean import load_and_clean_channels, load_and_clean_videos

    channels_df = load_and_clean_channels(channels_dir)
    videos_df = load_and_clean_videos(videos_dir)

    # Persist cleaned DataFrames for downstream consumers (e.g. reporting)
    processed_dir = ROOT_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    channels_df.to_parquet(processed_dir / "channels_clean.parquet", index=False)
    videos_df.to_parquet(processed_dir / "videos_clean.parquet", index=False)

    features_df = compute_features(channels_df, videos_df)
    save_features(features_df)

    return features_df
