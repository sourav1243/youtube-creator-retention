from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import ROOT_DIR, settings
from src.extraction.manifest import load_manifest, save_manifest
from src.extraction.youtube_client import QuotaExhausted, YouTubeClient, now_iso

logger = logging.getLogger(__name__)


def extract_channel_videos(
    channel_id: str,
    uploads_playlist_id: str,
    client: YouTubeClient,
    output_dir: Path,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    try:
        playlist_items = client.get_playlist_items(uploads_playlist_id, max_pages=max_pages)
    except QuotaExhausted:
        raise
    except Exception as e:
        logger.warning("  %s: playlistItems failed: %s", channel_id, e)
        return []

    video_ids = []
    for item in playlist_items:
        vid = item.get("contentDetails", {}).get("videoId")
        if vid:
            video_ids.append(vid)

    if not video_ids:
        logger.info("  %s: no videos found in uploads playlist", channel_id)
        return []

    try:
        videos = client.get_videos(video_ids)
    except QuotaExhausted:
        raise
    except Exception as e:
        logger.warning("  %s: videos.list failed: %s", channel_id, e)
        return []

    return videos


def extract_videos_tier_b(
    channels_with_playlists: list[tuple[str, str]],
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    client: YouTubeClient | None = None,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir or ROOT_DIR / "data" / "raw" / "videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path or ROOT_DIR / "data" / "raw" / "manifest.csv")

    client = client or YouTubeClient()
    if max_pages is None:
        max_pages = settings.extraction.max_pages_per_channel
    manifest = load_manifest(manifest_path)

    manifest_entries: list[dict[str, str]] = list(manifest.values())
    existing_done = {
        e["channel_id"] for e in manifest_entries if e.get("stage") == "videos" and e.get("status") == "done"
    }

    all_videos: list[dict[str, Any]] = []
    remaining = [(cid, pid) for cid, pid in channels_with_playlists if cid not in existing_done]

    logger.info("Extracting Tier B (videos) for %d channels...", len(remaining))

    for channel_id, playlist_id in remaining:
        logger.info("  Channel %s: extracting videos...", channel_id)
        try:
            videos = extract_channel_videos(channel_id, playlist_id, client, output_dir, max_pages=max_pages)
        except QuotaExhausted:
            logger.warning("  %s: quota exhausted during extraction, persisting checkpoint", channel_id)
            save_manifest(manifest_path, manifest_entries)
            raise

        if videos:
            video_path = output_dir / f"{channel_id}.json"
            with open(video_path, "w") as f:
                json.dump(videos, f, default=str)
            all_videos.extend(videos)
            logger.info("    -> %d videos saved", len(videos))

        status = "done" if videos else "empty"
        manifest_entries.append(
            {
                "channel_id": channel_id,
                "stage": "videos",
                "status": status,
                "fetched_at": now_iso(),
            }
        )

    save_manifest(manifest_path, manifest_entries)

    return all_videos
