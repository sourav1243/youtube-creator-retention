from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.extract_channels import extract_channels_tier_a, extract_channels_from_seed
from src.extraction.extract_videos import extract_channel_videos
from src.extraction.quota_planner import compute_quota_plan
from src.extraction.seed_channels import build_seed_list
from src.extraction.youtube_client import QuotaExhausted, YouTubeClient


@pytest.fixture
def sample_channel_response() -> dict:
    return {
        "kind": "youtube#channelListResponse",
        "items": [
            {
                "id": "UCtestchannelid1",
                "snippet": {
                    "title": "Test Channel 1",
                    "description": "A test channel",
                    "publishedAt": "2020-01-01T00:00:00Z",
                    "country": "US",
                },
                "statistics": {
                    "viewCount": "1000000",
                    "subscriberCount": "50000",
                    "hiddenSubscriberCount": False,
                    "videoCount": "200",
                },
                "contentDetails": {
                    "relatedPlaylists": {
                        "uploads": "UUtestchannelid1",
                        "likes": "",
                    }
                },
            }
        ],
    }


def test_seed_channels_validates_format(tmp_path: Path):
    output = tmp_path / "seed.csv"
    result = build_seed_list(output)
    assert len(result) > 0
    for entry in result:
        cid = entry["channel_id"]
        assert cid.startswith("UC")
        assert len(cid) == 24


def test_seed_channels_no_duplicates(tmp_path: Path):
    output = tmp_path / "seed.csv"
    result = build_seed_list(output)
    ids = [e["channel_id"] for e in result]
    assert len(ids) == len(set(ids))


def test_extract_channels_from_seed(tmp_path: Path):
    import csv

    seed_file = tmp_path / "seed.csv"
    with open(seed_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel_id", "name", "niche"])
        writer.writerow(["UCtestid111111111111111", "Test", "tech"])
        writer.writerow(["UCtestid222222222222222", "Test2", "tech"])

    ids = extract_channels_from_seed(str(seed_file))
    assert len(ids) == 2


def test_youtube_client_batch_ids():
    client = YouTubeClient(api_key="test-key")
    ids = list(range(120))
    batches = client.batch_ids(ids, max_per_call=50)
    assert len(batches) == 3
    assert len(batches[0]) == 50
    assert len(batches[1]) == 50
    assert len(batches[2]) == 20


@patch("src.extraction.youtube_client.requests.Session.get")
def test_quota_exhausted_raises_cleanly(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.json.return_value = {
        "error": {
            "errors": [{"reason": "quotaExceeded"}],
            "message": "Quota exceeded",
        }
    }
    mock_get.return_value = mock_resp

    client = YouTubeClient(api_key="test-key")
    with pytest.raises(QuotaExhausted):
        client.get_channels(["UCtestid111111111111111"])


def test_quota_planner_computes_correctly():
    plan = compute_quota_plan(n_channels=5000, n_tier_b=1000, avg_videos=200, daily_quota=10000)
    assert plan.total_cost_tier_a_only == 100
    assert plan.total_cost_full == 8100
    assert plan.days_needed_for_full == 1


@patch("src.extraction.youtube_client.YouTubeClient.get_playlist_items")
@patch("src.extraction.youtube_client.YouTubeClient.get_videos")
def test_extract_channel_videos(mock_get_videos, mock_get_playlist_items):
    mock_get_playlist_items.return_value = [
        {"contentDetails": {"videoId": "vid1"}},
        {"contentDetails": {"videoId": "vid2"}},
    ]
    mock_get_videos.return_value = [
        {"id": "vid1", "snippet": {"title": "Video 1"}},
        {"id": "vid2", "snippet": {"title": "Video 2"}},
    ]

    client = YouTubeClient(api_key="test-key")
    result = extract_channel_videos("UCtest", "UUtest", client, Path("/tmp"))
    assert len(result) == 2
