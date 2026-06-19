from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.features.engineer import compute_features


@pytest.fixture
def channels_df():
    return pd.DataFrame({
        "channel_id": ["UCactive", "UCinsufficient", "UCno_videos"],
        "title": ["Active Channel", "Insufficient", "No Videos"],
        "subscriber_count": [100000, 5000, 1000],
    })


@pytest.fixture
def videos_df():
    ref = datetime(2024, 6, 15, tzinfo=timezone.utc)
    rows = []
    # UCactive: 10 videos in last 30d, 20 videos in 31-90d (all within 90d window)
    # View counts scaled to days-since-published to keep vpd roughly constant
    for i in range(10):
        days_ago = i * 2 + 1  # 1, 3, 5, ... 19 days ago
        rows.append({
            "video_id": f"active_recent_{i}",
            "channel_id": "UCactive",
            "published_at": ref - pd.Timedelta(days=days_ago),
            "view_count": days_ago * 1000,  # ~1000 views/day
            "like_count": 50,
            "comment_count": 5,
            "comments_disabled": False,
            "duration_seconds": 300,
        })
    for i in range(20):
        days_ago = 35 + i * 2  # 35, 37, 39, ... 73 — all within 90d
        rows.append({
            "video_id": f"active_older_{i}",
            "channel_id": "UCactive",
            "published_at": ref - pd.Timedelta(days=days_ago),
            "view_count": days_ago * 1000,  # same ~1000 views/day
            "like_count": 50,
            "comment_count": 5,
            "comments_disabled": False,
            "duration_seconds": 300,
        })

    # UCinsufficient: only 1 video
    rows.append({
        "video_id": "insufficient_1",
        "channel_id": "UCinsufficient",
        "published_at": ref - pd.Timedelta(days=5),
        "view_count": 1000,
        "like_count": 50,
        "comment_count": 5,
        "comments_disabled": False,
        "duration_seconds": 300,
    })

    # UCno_videos: no videos

    return pd.DataFrame(rows)


def test_compute_features_active_channel(channels_df, videos_df):
    ref = datetime(2024, 6, 15, tzinfo=timezone.utc)
    features = compute_features(channels_df, videos_df, reference_date=ref)
    feat = features[features["channel_id"] == "UCactive"].iloc[0]

    # 10 videos in 30 days
    assert feat["upload_freq_30d"] == pytest.approx(10 / 30)
    # 30 videos in 90 days
    assert feat["upload_freq_90d"] == pytest.approx(30 / 90)
    # Frequency trend: (10/30) / (30/90) = 1.0
    assert feat["freq_trend_ratio"] == pytest.approx(1.0, rel=0.1)

    # Engagement rate should be > 0
    assert feat["avg_engagement_rate"] > 0
    assert feat["avg_engagement_rate"] <= 1.0

    # Days since last upload should be 1 (most recent video is 1 day ago)
    assert feat["days_since_last_upload"] == 1

    # Should not have insufficient_history
    assert not feat["insufficient_history"]


def test_compute_features_insufficient_history(channels_df, videos_df):
    ref = datetime(2024, 6, 15, tzinfo=timezone.utc)
    features = compute_features(channels_df, videos_df, reference_date=ref)
    feat = features[features["channel_id"] == "UCinsufficient"].iloc[0]
    assert feat["insufficient_history"]
    assert pd.isna(feat["momentum_ratio"])


def test_compute_features_no_videos(channels_df, videos_df):
    ref = datetime(2024, 6, 15, tzinfo=timezone.utc)
    features = compute_features(channels_df, videos_df, reference_date=ref)
    feat = features[features["channel_id"] == "UCno_videos"].iloc[0]
    assert feat["insufficient_history"]
    assert pd.isna(feat["upload_freq_30d"])


def test_momentum_ratio_steady(channels_df, videos_df):
    """Active channel has same views-per-day in both windows -> momentum ~1."""
    ref = datetime(2024, 6, 15, tzinfo=timezone.utc)
    features = compute_features(channels_df, videos_df, reference_date=ref)
    feat = features[features["channel_id"] == "UCactive"].iloc[0]
    assert feat["momentum_ratio"] is not None
    # Both windows have ~1000 views/day, so momentum should be approximately 1
    assert 0.8 < feat["momentum_ratio"] < 1.2
