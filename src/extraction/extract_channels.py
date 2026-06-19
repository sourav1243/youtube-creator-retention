from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.config import ROOT_DIR
from src.extraction.manifest import load_manifest, save_manifest
from src.extraction.youtube_client import YouTubeClient, now_iso

logger = logging.getLogger(__name__)


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

    manifest_entries: list[dict[str, str]] = list(manifest.values())
    existing_done = {
        e["channel_id"] for e in manifest_entries
        if e.get("stage") == "channels" and e.get("status") == "done"
    }

    all_channels: list[dict[str, Any]] = []
    remaining = [cid for cid in channel_ids if cid not in existing_done]

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
                save_manifest(manifest_path, manifest_entries)
                raise

    else:
        # Load channel data from previously saved JSON for already-done channels
        for json_file in sorted(output_dir.glob("*.json")):
            with open(json_file) as f:
                items = json.load(f)
            all_channels.extend(items)

    manifest_entries.sort(key=lambda x: (x["channel_id"], x["stage"]))
    save_manifest(manifest_path, manifest_entries)

    return all_channels


def extract_channels_from_seed(
    seed_csv: str | Path = ROOT_DIR / "data" / "raw" / "seed_channel_ids.csv",
    n_channels: int | None = None,
) -> list[str]:
    seed_csv = Path(seed_csv)
    channel_ids: list[str] = []
    if not seed_csv.exists():
        logger.warning("Seed CSV not found: %s", seed_csv)
        return []
    with open(seed_csv, newline="") as f:
        for row in csv.DictReader(f):
            channel_ids.append(row["channel_id"])

    n_channels = n_channels or len(channel_ids)
    return channel_ids[:n_channels]
