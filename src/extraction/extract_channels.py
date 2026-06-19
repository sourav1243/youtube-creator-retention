from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import ROOT_DIR, settings
from src.extraction.youtube_client import YouTubeClient, now_iso

logger = logging.getLogger(__name__)


MANIFEST_HEADER = ["channel_id", "stage", "status", "fetched_at"]


def load_manifest(manifest_path: str | Path) -> dict[str, dict[str, str]]:
    manifest_path = Path(manifest_path)
    manifest: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest[row["channel_id"]] = row
    return manifest


def save_manifest(manifest_path: str | Path, entries: list[dict[str, str]]):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(entries)


def extract_channels_tier_a(
    channel_ids: list[str],
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    client: YouTubeClient | None = None,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir or ROOT_DIR / "data" / "raw" / "channels")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path or ROOT_DIR / "data" / "raw" / "manifest.csv")

    client = client or YouTubeClient()
    manifest = load_manifest(manifest_path)

    manifest_entries: list[dict[str, str]] = []
    all_channels: list[dict[str, Any]] = []

    # Include channels already done in manifest
    for cid, entry in manifest.items():
        if entry.get("stage") == "channels" and entry.get("status") == "done":
            manifest_entries.append(entry)
            logger.debug("Skipping already-fetched channel: %s", cid)

    remaining = [cid for cid in channel_ids if cid not in {e["channel_id"] for e in manifest_entries}]

    if remaining:
        logger.info("Extracting Tier A (channels.list) for %d channels...", len(remaining))

        for batch_idx, batch in enumerate(client.batch_ids(remaining)):
            batch_id = f"channels_batch_{batch_idx:04d}"
            try:
                items = client.get_channels(batch)
                batch_path = output_dir / f"{batch_id}.json"
                with open(batch_path, "w") as f:
                    json.dump(items, f, default=str)
                logger.info("  Batch %s: %d channels saved", batch_id, len(items))

                for item in items:
                    cid = item.get("id", "")
                    manifest_entries.append({
                        "channel_id": cid,
                        "stage": "channels",
                        "status": "done",
                        "fetched_at": now_iso(),
                    })
                    all_channels.append(item)
            except Exception as e:
                logger.error("  Batch %s failed: %s", batch_id, e)
                for cid in batch:
                    manifest_entries.append({
                        "channel_id": cid,
                        "stage": "channels",
                        "status": "failed",
                        "fetched_at": now_iso(),
                    })
                raise

    # Re-sort manifest to maintain stable order
    manifest_entries.sort(key=lambda x: x["channel_id"])
    save_manifest(manifest_path, manifest_entries)

    return all_channels


def extract_channels_from_seed(
    seed_csv: str | Path = ROOT_DIR / "data" / "raw" / "seed_channel_ids.csv",
    n_channels: int | None = None,
) -> list[str]:
    seed_csv = Path(seed_csv)
    channel_ids: list[str] = []
    with open(seed_csv) as f:
        for row in csv.DictReader(f):
            channel_ids.append(row["channel_id"])

    n_channels = n_channels or len(channel_ids)
    return channel_ids[:n_channels]
