from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.clean import (
    cap_outliers,
    cast_numeric,
    drop_duplicate_videos,
)


def test_cast_numeric_coerces_bad_values():
    s = pd.Series(["100", "200", "bad", "300"])
    result = cast_numeric(s, "test")
    assert result.isna().sum() == 1  # only "bad" gets coerced
    assert result.notna().sum() == 3


def test_cast_numeric_all_valid():
    s = pd.Series(["100", "200", "300"])
    result = cast_numeric(s, "test")
    assert result.isna().sum() == 0


def test_cap_outliers():
    np.random.seed(42)
    s = pd.Series(np.random.exponential(scale=100, size=1000))
    # Add extreme outliers
    s.iloc[0] = 1_000_000
    s.iloc[1] = 2_000_000

    result = cap_outliers(s, "test")
    assert result.max() < 500_000  # Should be capped well below the outliers
    assert result.isna().sum() == 0


def test_cap_outliers_does_not_affect_most_data():
    s = pd.Series(range(1, 101))
    result = cap_outliers(s, "test", percentile=0.95)
    # Only top 5 values max should be capped
    assert result.max() <= 100


def test_drop_duplicate_videos():
    df = pd.DataFrame(
        {
            "video_id": ["a", "b", "a", "c", "b"],
            "view_count": [100, 200, 100, 300, 200],
        }
    )
    result = drop_duplicate_videos(df)
    assert len(result) == 3  # a, b, c


def test_drop_duplicate_videos_no_duplicates():
    df = pd.DataFrame(
        {
            "video_id": ["a", "b", "c"],
            "view_count": [100, 200, 300],
        }
    )
    result = drop_duplicate_videos(df)
    assert len(result) == 3
