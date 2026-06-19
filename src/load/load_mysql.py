from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from isodate import parse_duration
from sqlalchemy import create_engine, text

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


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value) if isinstance(value, bool) else None


def init_sqlite_db(engine):
    schema_sql = """
    CREATE TABLE IF NOT EXISTS channels (
        channel_id          TEXT PRIMARY KEY,
        title               TEXT NOT NULL,
        description         TEXT,
        country             TEXT,
        published_at        TEXT,
        uploads_playlist_id TEXT,
        fetched_at          TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS channel_snapshots (
        snapshot_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id          TEXT NOT NULL,
        snapshot_date       TEXT NOT NULL,
        subscriber_count    INTEGER NULL,
        subscriber_hidden   BOOLEAN NOT NULL DEFAULT 0,
        view_count_total    INTEGER NULL,
        video_count         INTEGER NULL,
        UNIQUE (channel_id, snapshot_date),
        FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
    );

    CREATE TABLE IF NOT EXISTS videos (
        video_id            TEXT PRIMARY KEY,
        channel_id          TEXT NOT NULL,
        published_at        TEXT NOT NULL,
        duration_seconds    INTEGER,
        view_count          INTEGER NULL,
        like_count          INTEGER NULL,
        comment_count       INTEGER NULL,
        comments_disabled   BOOLEAN NOT NULL DEFAULT 0,
        fetched_at          TEXT NOT NULL,
        FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
    );

    CREATE TABLE IF NOT EXISTS creator_features (
        channel_id              TEXT PRIMARY KEY,
        computed_at             TEXT NOT NULL,
        upload_freq_30d         REAL,
        upload_freq_90d         REAL,
        freq_trend_ratio        REAL,
        momentum_ratio          REAL,
        avg_engagement_rate     REAL,
        days_since_last_upload  INTEGER,
        upload_regularity       REAL,
        duration_trend          REAL,
        insufficient_history    BOOLEAN NOT NULL DEFAULT 0,
        FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
    );

    CREATE TABLE IF NOT EXISTS creator_clusters (
        channel_id          TEXT PRIMARY KEY,
        model_version       TEXT NOT NULL,
        algorithm           TEXT NOT NULL DEFAULT 'kmeans',
        cluster_id          INTEGER NOT NULL,
        cluster_label       TEXT NOT NULL,
        risk_flag           TEXT CHECK(risk_flag IN ('Healthy','Watch','At-Risk','Unscored')) NOT NULL DEFAULT 'Unscored',
        risk_score          REAL,
        confidence          REAL,
        distance_to_centroid REAL,
        scored_at           TEXT NOT NULL,
        FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
    );
    """
    with engine.begin() as conn:
        for statement in schema_sql.strip().split(";"):
            if statement.strip():
                conn.execute(text(statement))


def get_engine():
    mysql_dsn = settings.mysql.dsn
    try:
        # Try creating MySQL engine and testing connection
        engine = create_engine(mysql_dsn, connect_args={"connect_timeout": 3})
        # Smoke test connection
        with engine.connect():
            pass
        logger.info("Connected to MySQL database")
        return engine
    except Exception as e:
        db_dir = ROOT_DIR / "data" / "processed"
        db_dir.mkdir(parents=True, exist_ok=True)
        sqlite_path = db_dir / "youtube_creator_retention.db"
        logger.warning("MySQL connection failed (%s). Falling back to SQLite database at: %s", e, sqlite_path)
        sqlite_dsn = f"sqlite:///{sqlite_path.as_posix()}"
        engine = create_engine(sqlite_dsn)
        init_sqlite_db(engine)
        return engine


def parse_channel_json(filepath: Path) -> list[dict[str, Any]]:
    with open(filepath, encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    now = datetime.now(UTC)
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

        rows.append(
            {
                "channel_id": cid,
                "title": snippet.get("title", ""),
                "description": snippet.get("description"),
                "country": snippet.get("country"),
                "published_at": published_at,
                "uploads_playlist_id": content.get("relatedPlaylists", {}).get("uploads"),
                "fetched_at": now,
            }
        )

        hidden_sub = stats.get("hiddenSubscriberCount")
        rows.append(
            {
                "_snapshot": True,
                "channel_id": cid,
                "snapshot_date": now.date(),
                "subscriber_count": _safe_int(stats.get("subscriberCount")),
                "subscriber_hidden": _safe_bool(hidden_sub),
                "view_count_total": _safe_int(stats.get("viewCount")),
                "video_count": _safe_int(stats.get("videoCount")),
            }
        )

    return rows


def parse_video_json(filepath: Path) -> list[dict[str, Any]]:
    with open(filepath, encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    now = datetime.now(UTC)
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

        rows.append(
            {
                "video_id": item.get("id", ""),
                "channel_id": cid,
                "published_at": published_at,
                "duration_seconds": _parse_iso8601_duration(content.get("duration")),
                "view_count": _safe_int(stats.get("viewCount")),
                "like_count": _safe_int(stats.get("likeCount")),
                "comment_count": comment_count,
                "comments_disabled": comments_disabled,
                "fetched_at": now,
            }
        )

    return rows


def upsert_channels(engine, rows: list[dict[str, Any]]):
    channel_rows = [r for r in rows if not r.get("_snapshot")]
    if not channel_rows:
        return
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            query = """
                INSERT INTO channels (channel_id, title, description, country, published_at, uploads_playlist_id, fetched_at)
                VALUES (:channel_id, :title, :description, :country, :published_at, :uploads_playlist_id, :fetched_at)
                ON CONFLICT(channel_id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    country = excluded.country,
                    published_at = excluded.published_at,
                    uploads_playlist_id = excluded.uploads_playlist_id,
                    fetched_at = excluded.fetched_at
            """
        else:
            query = """
                INSERT INTO channels (channel_id, title, description, country, published_at, uploads_playlist_id, fetched_at)
                VALUES (:channel_id, :title, :description, :country, :published_at, :uploads_playlist_id, :fetched_at)
                ON DUPLICATE KEY UPDATE
                    title = VALUES(title),
                    description = VALUES(description),
                    country = VALUES(country),
                    published_at = VALUES(published_at),
                    uploads_playlist_id = VALUES(uploads_playlist_id),
                    fetched_at = VALUES(fetched_at)
            """
        conn.execute(text(query), channel_rows)
    logger.info("Upserted %d channels", len(channel_rows))


def upsert_channel_snapshots(engine, rows: list[dict[str, Any]]):
    snapshot_rows = [r for r in rows if r.get("_snapshot")]
    if not snapshot_rows:
        return
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            query = """
                INSERT INTO channel_snapshots (channel_id, snapshot_date, subscriber_count, subscriber_hidden, view_count_total, video_count)
                VALUES (:channel_id, :snapshot_date, :subscriber_count, :subscriber_hidden, :view_count_total, :video_count)
                ON CONFLICT(channel_id, snapshot_date) DO UPDATE SET
                    subscriber_count = excluded.subscriber_count,
                    subscriber_hidden = excluded.subscriber_hidden,
                    view_count_total = excluded.view_count_total,
                    video_count = excluded.video_count
            """
        else:
            query = """
                INSERT INTO channel_snapshots (channel_id, snapshot_date, subscriber_count, subscriber_hidden, view_count_total, video_count)
                VALUES (:channel_id, :snapshot_date, :subscriber_count, :subscriber_hidden, :view_count_total, :video_count)
                ON DUPLICATE KEY UPDATE
                    subscriber_count = VALUES(subscriber_count),
                    subscriber_hidden = VALUES(subscriber_hidden),
                    view_count_total = VALUES(view_count_total),
                    video_count = VALUES(video_count)
            """
        conn.execute(text(query), snapshot_rows)
    logger.info("Upserted %d channel snapshots", len(snapshot_rows))


def upsert_videos(engine, rows: list[dict[str, Any]]):
    if not rows:
        return
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            query = """
                INSERT INTO videos (video_id, channel_id, published_at, duration_seconds, view_count, like_count, comment_count, comments_disabled, fetched_at)
                VALUES (:video_id, :channel_id, :published_at, :duration_seconds, :view_count, :like_count, :comment_count, :comments_disabled, :fetched_at)
                ON CONFLICT(video_id) DO UPDATE SET
                    view_count = excluded.view_count,
                    like_count = excluded.like_count,
                    comment_count = excluded.comment_count,
                    comments_disabled = excluded.comments_disabled,
                    fetched_at = excluded.fetched_at
            """
        else:
            query = """
                INSERT INTO videos (video_id, channel_id, published_at, duration_seconds, view_count, like_count, comment_count, comments_disabled, fetched_at)
                VALUES (:video_id, :channel_id, :published_at, :duration_seconds, :view_count, :like_count, :comment_count, :comments_disabled, :fetched_at)
                ON DUPLICATE KEY UPDATE
                    view_count = VALUES(view_count),
                    like_count = VALUES(like_count),
                    comment_count = VALUES(comment_count),
                    comments_disabled = VALUES(comments_disabled),
                    fetched_at = VALUES(fetched_at)
            """
        conn.execute(text(query), rows)
    logger.info("Upserted %d videos", len(rows))


VALID_TABLES = {"channels", "channel_snapshots", "videos", "creator_features", "creator_clusters"}


def reconciliation_check(engine, manifest_rows: int, table: str = "channels") -> bool:
    if table not in VALID_TABLES:
        logger.error("Invalid table name for reconciliation: %s", table)
        return False
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            db_count = result.scalar()
            match = db_count == manifest_rows
            if not match:
                logger.warning(
                    "Reconciliation: %s — DB has %d rows, manifest shows %d (delta: %d)",
                    table,
                    db_count,
                    manifest_rows,
                    abs(db_count - manifest_rows),
                )
            else:
                logger.info("Reconciliation OK: %s — %d rows match", table, db_count)
            return match
    except Exception as e:
        logger.error("Reconciliation failed: %s", e)
        return False


def upsert_features(engine, features_parquet: str | Path | None = None):
    import pandas as pd

    parquet_path = Path(features_parquet or ROOT_DIR / "data" / "processed" / "creator_features.parquet")
    if not parquet_path.exists():
        logger.warning("Features parquet not found: %s", parquet_path)
        return 0
    df = pd.read_parquet(parquet_path)
    row_dicts = []
    for _, row in df.iterrows():
        row_dict = {
            k: (v.to_pydatetime() if hasattr(v, "to_pydatetime") else (None if pd.isna(v) else v))
            for k, v in row.to_dict().items()
        }
        row_dict.setdefault("freq_trend_ratio", None)
        row_dicts.append(row_dict)

    if engine.dialect.name == "sqlite":
        query = """
            INSERT INTO creator_features (channel_id, computed_at, upload_freq_30d, upload_freq_90d, freq_trend_ratio,
                momentum_ratio, avg_engagement_rate, days_since_last_upload, upload_regularity, duration_trend, insufficient_history)
            VALUES (:channel_id, :computed_at, :upload_freq_30d, :upload_freq_90d, :freq_trend_ratio,
                :momentum_ratio, :avg_engagement_rate, :days_since_last_upload, :upload_regularity, :duration_trend, :insufficient_history)
            ON CONFLICT(channel_id) DO UPDATE SET
                computed_at = excluded.computed_at,
                upload_freq_30d = excluded.upload_freq_30d,
                upload_freq_90d = excluded.upload_freq_90d,
                freq_trend_ratio = excluded.freq_trend_ratio,
                momentum_ratio = excluded.momentum_ratio,
                avg_engagement_rate = excluded.avg_engagement_rate,
                days_since_last_upload = excluded.days_since_last_upload,
                upload_regularity = excluded.upload_regularity,
                duration_trend = excluded.duration_trend,
                insufficient_history = excluded.insufficient_history
        """
    else:
        query = """
            INSERT INTO creator_features (channel_id, computed_at, upload_freq_30d, upload_freq_90d, freq_trend_ratio,
                momentum_ratio, avg_engagement_rate, days_since_last_upload, upload_regularity, duration_trend, insufficient_history)
            VALUES (:channel_id, :computed_at, :upload_freq_30d, :upload_freq_90d, :freq_trend_ratio,
                :momentum_ratio, :avg_engagement_rate, :days_since_last_upload, :upload_regularity, :duration_trend, :insufficient_history)
            ON DUPLICATE KEY UPDATE
                computed_at = VALUES(computed_at),
                upload_freq_30d = VALUES(upload_freq_30d),
                upload_freq_90d = VALUES(upload_freq_90d),
                freq_trend_ratio = VALUES(freq_trend_ratio),
                momentum_ratio = VALUES(momentum_ratio),
                avg_engagement_rate = VALUES(avg_engagement_rate),
                days_since_last_upload = VALUES(days_since_last_upload),
                upload_regularity = VALUES(upload_regularity),
                duration_trend = VALUES(duration_trend),
                insufficient_history = VALUES(insufficient_history)
        """
    with engine.begin() as conn:
        conn.execute(text(query), row_dicts)


def upsert_clusters(engine, clusters_parquet: str | Path | None = None):
    import pandas as pd

    parquet_path = Path(clusters_parquet or ROOT_DIR / "data" / "processed" / "creator_clusters.parquet")
    if not parquet_path.exists():
        logger.warning("Clusters parquet not found: %s", parquet_path)
        return 0
    df = pd.read_parquet(parquet_path)
    row_dicts = []
    for _, row in df.iterrows():
        row_dict = {
            k: (v.to_pydatetime() if hasattr(v, "to_pydatetime") else (None if pd.isna(v) else v))
            for k, v in row.to_dict().items()
        }
        row_dict["algorithm"] = str(row.get("model_version", "kmeans_v1")).split("_")[0] if row.get("model_version") else "kmeans"
        row_dicts.append(row_dict)

    if engine.dialect.name == "sqlite":
        query = """
            INSERT INTO creator_clusters (channel_id, model_version, algorithm, cluster_id,
                cluster_label, risk_flag, risk_score, confidence, distance_to_centroid, scored_at)
            VALUES (:channel_id, :model_version, :algorithm, :cluster_id,
                :cluster_label, :risk_flag, :risk_score, :confidence, :distance_to_centroid, :scored_at)
            ON CONFLICT(channel_id) DO UPDATE SET
                model_version = excluded.model_version,
                algorithm = excluded.algorithm,
                cluster_id = excluded.cluster_id,
                cluster_label = excluded.cluster_label,
                risk_flag = excluded.risk_flag,
                risk_score = excluded.risk_score,
                confidence = excluded.confidence,
                distance_to_centroid = excluded.distance_to_centroid,
                scored_at = excluded.scored_at
        """
    else:
        query = """
            INSERT INTO creator_clusters (channel_id, model_version, algorithm, cluster_id,
                cluster_label, risk_flag, risk_score, confidence, distance_to_centroid, scored_at)
            VALUES (:channel_id, :model_version, :algorithm, :cluster_id,
                :cluster_label, :risk_flag, :risk_score, :confidence, :distance_to_centroid, :scored_at)
            ON DUPLICATE KEY UPDATE
                model_version = VALUES(model_version),
                algorithm = VALUES(algorithm),
                cluster_id = VALUES(cluster_id),
                cluster_label = VALUES(cluster_label),
                risk_flag = VALUES(risk_flag),
                risk_score = VALUES(risk_score),
                confidence = VALUES(confidence),
                distance_to_centroid = VALUES(distance_to_centroid),
                scored_at = VALUES(scored_at)
        """
    with engine.begin() as conn:
        conn.execute(text(query), row_dicts)
    logger.info("Upserted %d cluster rows", len(df))
    return len(df)


def load_all(engine=None, channels_dir: str | Path | None = None, videos_dir: str | Path | None = None):
    own_engine = engine is None
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
                upsert_channels(engine, rows)
                upsert_channel_snapshots(engine, rows)
                total_channels += len([r for r in rows if not r.get("_snapshot")])

    if videos_dir.exists():
        for json_file in sorted(videos_dir.glob("*.json")):
            rows = parse_video_json(json_file)
            if rows:
                upsert_videos(engine, rows)
                total_videos += len(rows)

    reconciliation_check(engine, total_channels, "channels")
    reconciliation_check(engine, total_videos, "videos")

    # Load features and clusters
    total_features = upsert_features(engine)
    total_clusters = upsert_clusters(engine)

    if own_engine:
        engine.dispose()

    logger.info(
        "Load complete: %d channels, %d videos, %d features, %d clusters",
        total_channels,
        total_videos,
        total_features,
        total_clusters,
    )
    return total_channels, total_videos
