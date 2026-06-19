from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_HEADER = ["channel_id", "stage", "status", "fetched_at"]


def load_manifest(manifest_path: str | Path) -> dict[str, dict[str, str]]:
    manifest_path = Path(manifest_path)
    manifest: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != MANIFEST_HEADER:
                logger.warning("Manifest header mismatch: expected %s, got %s", MANIFEST_HEADER, reader.fieldnames)
            for row in reader:
                key = f"{row['channel_id']}:{row['stage']}"
                manifest[key] = row
    return manifest


def save_manifest(manifest_path: str | Path, entries: list[dict[str, str]]) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(entries)
