from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from isodate import parse_duration
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from src.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)


def _parse_iso8601_duration(duration_str: str | None) -> int | None:
    if not duration_str:
        return None
    try:
        td = parse_duration(duration_str)
        return int(td.total_seconds())
    except Exception:
        logger.warning("Could not parse duration: %s", duration_str)
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return False


def get_engine():
    return create_engine(settings.mysql.dsn)


def parse_channel_json(filepath: Path) -> list[dict[str, Any]]:
    with open(filepath) as f:
        items = json.load(f)

    rows = []
    now = datetime.now(timezone.utc)
    for item in items:
        cid = item.get("id", "")
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        published_at = None
        pub_str = snippet.get("publishedAt")
        if pub_str:
            try:
                published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        rows.append({
            "channel_id": cid,
            "title": snippet.get("title", ""),
            "description": snippet.get("description"),
            "country": snippet.get("country"),
            "published_at": published_at,
            "uploads_playlist_id": content.get("relatedPlaylists", {}).get("uploads"),
            "fetched_at": now,
        })

        # Also create a snapshot entry
        rows.append({
            "_snapshot": True,
            "channel_id": cid,
            "snapshot_date": now.date(),
            "subscriber_count": _safe_int(stats.get("subscriberCount")),
            "subscriber_hidden": _safe_bool(stats.get("hiddenSubscriberCount", False)),
            "view_count_total": _safe_int(stats.get("viewCount")),
            "video_count": _safe_int(stats.get("videoCount")),
        })

    return rows


def parse_video_json(filepath: Path) -> list[dict[str, Any]]:
    with open(filepath) as f:
        items = json.load(f)

    rows = []
    now = datetime.now(timezone.utc)
    for item in items:
        cid = item.get("snippet", {}).get("channelId", "")
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        published_at = None
        pub_str = snippet.get("publishedAt")
        if pub_str:
            try:
                published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        comment_count = _safe_int(stats.get("commentCount"))
        comments_disabled = "commentCount" not in stats

        rows.append({
            "video_id": item.get("id", ""),
            "channel_id": cid,
            "published_at": published_at,
            "duration_seconds": _parse_iso8601_duration(content.get("duration")),
            "view_count": _safe_int(stats.get("viewCount")),
            "like_count": _safe_int(stats.get("likeCount")),
            "comment_count": comment_count,
            "comments_disabled": comments_disabled,
            "fetched_at": now,
        })

    return rows


def upsert_channels(engine, rows: list[dict[str, Any]]):
    with engine.begin() as conn:
        for row in rows:
            if row.get("_snapshot"):
                continue
            conn.execute(
                text("""
                    INSERT INTO channels (channel_id, title, description, country, published_at, uploads_playlist_id, fetched_at)
                    VALUES (:channel_id, :title, :description, :country, :published_at, :uploads_playlist_id, :fetched_at)
                    ON DUPLICATE KEY UPDATE
                        title = VALUES(title),
                        description = VALUES(description),
                        country = VALUES(country),
                        published_at = VALUES(published_at),
                        uploads_playlist_id = VALUES(uploads_playlist_id),
                        fetched_at = VALUES(fetched_at)
                """),
                row,
            )
    logger.info("Upserted %d channels", len([r for r in rows if not r.get("_snapshot")]))


def upsert_channel_snapshots(engine, rows: list[dict[str, Any]]):
    with engine.begin() as conn:
        for row in rows:
            if not row.get("_snapshot"):
                continue
            conn.execute(
                text("""
                    INSERT INTO channel_snapshots (channel_id, snapshot_date, subscriber_count, subscriber_hidden, view_count_total, video_count)
                    VALUES (:channel_id, :snapshot_date, :subscriber_count, :subscriber_hidden, :view_count_total, :video_count)
                    ON DUPLICATE KEY UPDATE
                        subscriber_count = VALUES(subscriber_count),
                        subscriber_hidden = VALUES(subscriber_hidden),
                        view_count_total = VALUES(view_count_total),
                        video_count = VALUES(video_count)
                """),
                row,
            )
    logger.info("Upserted %d channel snapshots", len([r for r in rows if r.get("_snapshot")]))


def upsert_videos(engine, rows: list[dict[str, Any]]):
    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text("""
                    INSERT INTO videos (video_id, channel_id, published_at, duration_seconds, view_count, like_count, comment_count, comments_disabled, fetched_at)
                    VALUES (:video_id, :channel_id, :published_at, :duration_seconds, :view_count, :like_count, :comment_count, :comments_disabled, :fetched_at)
                    ON DUPLICATE KEY UPDATE
                        view_count = VALUES(view_count),
                        like_count = VALUES(like_count),
                        comment_count = VALUES(comment_count),
                        comments_disabled = VALUES(comments_disabled),
                        fetched_at = VALUES(fetched_at)
                """),
                row,
            )
    logger.info("Upserted %d videos", len(rows))


def reconciliation_check(engine, manifest_rows: int, label: str = "channels") -> bool:
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {label}"))
            db_count = result.scalar()
            match = db_count == manifest_rows
            if not match:
                logger.warning(
                    "Reconciliation: %s — DB has %d rows, manifest shows %d (delta: %d)",
                    label, db_count, manifest_rows, abs(db_count - manifest_rows),
                )
            else:
                logger.info("Reconciliation OK: %s — %d rows match", label, db_count)
            return match
    except Exception as e:
        logger.error("Reconciliation failed: %s", e)
        return False


def load_all(engine=None, channels_dir: str | Path | None = None, videos_dir: str | Path | None = None):
    if engine is None:
        engine = get_engine()

    channels_dir = Path(channels_dir or ROOT_DIR / "data" / "raw" / "channels")
    videos_dir = Path(videos_dir or ROOT_DIR / "data" / "raw" / "videos")

    total_channels = 0
    total_videos = 0

    if channels_dir.exists():
        for json_file in sorted(channels_dir.glob("*.json")):
            rows = parse_channel_json(json_file)
            if rows:
                channel_rows = [r for r in rows if not r.get("_snapshot")]
                snapshot_rows = [r for r in rows if r.get("_snapshot")]
                upsert_channels(engine, rows)
                upsert_channel_snapshots(engine, rows)
                total_channels += len(channel_rows)
                reconciliation_check(engine, total_channels, "channels")

    if videos_dir.exists():
        for json_file in sorted(videos_dir.glob("*.json")):
            rows = parse_video_json(json_file)
            if rows:
                upsert_videos(engine, rows)
                total_videos += len(rows)
                reconciliation_check(engine, total_videos, "videos")

    logger.info("Load complete: %d channels, %d videos", total_channels, total_videos)
    return total_channels, total_videos
