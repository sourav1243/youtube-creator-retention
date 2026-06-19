from __future__ import annotations

import json
from pathlib import Path

from src.load.load_mysql import parse_channel_json, parse_video_json, _parse_iso8601_duration, _safe_int


def test_safe_int_with_none():
    assert _safe_int(None) is None


def test_safe_int_with_string():
    assert _safe_int("12345") == 12345


def test_safe_int_with_large_number():
    assert _safe_int("99999999999") == 99999999999


def test_parse_iso8601_duration():
    assert _parse_iso8601_duration("PT4M13S") == 253
    assert _parse_iso8601_duration("PT1H2M30S") == 3750
    assert _parse_iso8601_duration("P1DT2H") == 93600
    assert _parse_iso8601_duration(None) is None


def test_parse_channel_json(tmp_path: Path):
    data = [
        {
            "id": "UCtest123",
            "snippet": {
                "title": "Test Channel",
                "description": "A test",
                "publishedAt": "2020-01-15T00:00:00Z",
                "country": "US",
            },
            "statistics": {
                "viewCount": "500000",
                "subscriberCount": "25000",
                "hiddenSubscriberCount": False,
                "videoCount": "100",
            },
            "contentDetails": {
                "relatedPlaylists": {"uploads": "UUtest123", "likes": ""},
            },
        },
        {
            "id": "UCtest456",
            "snippet": {
                "title": "Test Channel 2",
                "description": None,
                "publishedAt": "2019-06-01T12:00:00Z",
                "country": None,
            },
            "statistics": {
                "viewCount": "1000000",
                "subscriberCount": None,
                "hiddenSubscriberCount": True,
                "videoCount": "50",
            },
            "contentDetails": {
                "relatedPlaylists": {"uploads": "UUtest456"},
            },
        },
    ]

    filepath = tmp_path / "channels.json"
    with open(filepath, "w") as f:
        json.dump(data, f)

    rows = parse_channel_json(filepath)
    # 2 channels + 2 snapshots = 4 rows
    assert len(rows) == 4

    channel_rows = [r for r in rows if not r.get("_snapshot")]
    assert len(channel_rows) == 2
    assert channel_rows[0]["channel_id"] == "UCtest123"
    assert channel_rows[0]["title"] == "Test Channel"
    assert channel_rows[0]["country"] == "US"

    snapshot_rows = [r for r in rows if r.get("_snapshot")]
    assert len(snapshot_rows) == 2
    # Channel with hiddenSubscriberCount=True should have subscriber_count=None
    assert snapshot_rows[1]["channel_id"] == "UCtest456"
    assert snapshot_rows[1]["subscriber_hidden"] is True


def test_parse_video_json(tmp_path: Path):
    data = [
        {
            "id": "vid001",
            "snippet": {
                "channelId": "UCtest123",
                "publishedAt": "2023-01-10T00:00:00Z",
            },
            "statistics": {
                "viewCount": "10000",
                "likeCount": "500",
                "commentCount": "50",
            },
            "contentDetails": {
                "duration": "PT10M30S",
            },
        },
        {
            "id": "vid002",
            "snippet": {
                "channelId": "UCtest123",
                "publishedAt": "2023-02-15T00:00:00Z",
            },
            "statistics": {
                "viewCount": "5000",
                "likeCount": "200",
            },
            "contentDetails": {
                "duration": "PT5M",
            },
        },
    ]

    filepath = tmp_path / "videos.json"
    with open(filepath, "w") as f:
        json.dump(data, f)

    rows = parse_video_json(filepath)
    assert len(rows) == 2
    assert rows[0]["video_id"] == "vid001"
    assert rows[0]["duration_seconds"] == 630
    assert rows[0]["comments_disabled"] is False
    assert rows[0]["comment_count"] == 50

    # Second video has no commentCount -> comments_disabled should be True
    assert rows[1]["comments_disabled"] is True
    assert rows[1]["comment_count"] is None
