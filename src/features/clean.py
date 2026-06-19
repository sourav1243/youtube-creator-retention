from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from isodate import parse_duration

from src.config import ROOT_DIR, settings

warnings.filterwarnings("ignore", message="Downcasting behavior in Series", category=FutureWarning)

logger = logging.getLogger(__name__)


def cast_numeric(series: pd.Series, name: str) -> pd.Series:
    original_count = series.notna().sum()
    result = pd.to_numeric(series, errors="coerce")
    dropped = original_count - result.notna().sum()
    if dropped > 0:
        logger.warning("  %s: coerced %d values to NaN (non-numeric)", name, dropped)
    if result.notna().sum() == 0 and original_count > 0:
        logger.warning("  %s: ALL values coerced to NaN — column may be corrupt", name)
    return result


def cap_outliers(
    series: pd.Series,
    name: str,
    percentile: float = 0.99,
    enabled: bool = True,
) -> pd.Series:
    if not enabled:
        return series
    cap = series.quantile(percentile)
    capped_count = (series > cap).sum()
    if capped_count > 0:
        logger.info("  %s: capped %d values at %.2f (P%.0f)", name, capped_count, cap, percentile * 100)
    return series.clip(upper=cap)


def drop_duplicate_videos(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["video_id"])
    after = len(df)
    dropped = before - after
    if dropped > 0:
        logger.warning("Dropped %d duplicate video rows", dropped)
    return df


def _safe_load_json(filepath: Path) -> list[dict[str, Any]]:
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Skipping corrupt JSON file %s: %s", filepath, e)
        return []


def load_and_clean_channels(channels_dir: str | Path | None = None) -> pd.DataFrame:
    channels_dir = Path(channels_dir or ROOT_DIR / "data" / "raw" / "channels")
    records: list[dict[str, Any]] = []

    if not channels_dir.exists():
        logger.warning("Channels directory not found: %s", channels_dir)
        return pd.DataFrame()

    for json_file in sorted(channels_dir.glob("*.json")):
        items = _safe_load_json(json_file)
        for item in items:
            cid = item.get("id", "")
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            records.append(
                {
                    "channel_id": cid,
                    "title": snippet.get("title"),
                    "subscriber_count": stats.get("subscriberCount"),
                    "view_count_total": stats.get("viewCount"),
                    "video_count": stats.get("videoCount"),
                    "hidden_subscriber": stats.get("hiddenSubscriberCount", None),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    cap_enabled = settings.cleaning.outlier_cap_flag
    for col in ["subscriber_count", "view_count_total", "video_count"]:
        df[col] = cast_numeric(df[col], col)
        df[col] = cap_outliers(df[col], col, enabled=cap_enabled)

    return df


def load_and_clean_videos(videos_dir: str | Path | None = None) -> pd.DataFrame:
    videos_dir = Path(videos_dir or ROOT_DIR / "data" / "raw" / "videos")
    records: list[dict[str, Any]] = []

    if not videos_dir.exists():
        logger.warning("Videos directory not found: %s", videos_dir)
        return pd.DataFrame()

    for json_file in sorted(videos_dir.glob("*.json")):
        items = _safe_load_json(json_file)
        for item in items:
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})
            cid = snippet.get("channelId", "")

            dur_str = content.get("duration", "")
            duration_s = None
            if dur_str:
                try:
                    duration_s = int(parse_duration(dur_str).total_seconds())
                except Exception:
                    pass

            records.append(
                {
                    "video_id": item.get("id", ""),
                    "channel_id": cid,
                    "published_at": snippet.get("publishedAt"),
                    "view_count": stats.get("viewCount"),
                    "like_count": stats.get("likeCount"),
                    "comment_count": stats.get("commentCount"),
                    "comments_disabled": "commentCount" not in stats,
                    "duration_seconds": duration_s,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = drop_duplicate_videos(df)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    for col in ["view_count", "like_count", "comment_count"]:
        df[col] = cast_numeric(df[col], col)

    df["duration_seconds"] = cast_numeric(df["duration_seconds"], "duration_seconds")

    return df
