from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from src.config import settings
from src.modeling.cluster import (
    FEATURE_COLUMNS,
    assign_risk_flags,
    compute_direct_risk_score,
    fit_and_persist,
    label_clusters_by_risk,
)


def test_clustering_pipeline_deterministic():
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        n_features=len(FEATURE_COLUMNS),
        random_state=42,
        cluster_std=2.0,
    )

    df = pd.DataFrame(
        X,
        columns=FEATURE_COLUMNS,
    )
    df["channel_id"] = [f"UC{i:010d}" for i in range(300)]
    df["computed_at"] = "2024-06-15"
    df["insufficient_history"] = False

    model, scaler, scored_df, result, _ = fit_and_persist(
        df,
        random_state=settings.clustering.random_state,
    )

    assert model.n_clusters >= 3
    assert model.n_clusters <= 5
    assert "cluster_id" in scored_df.columns
    assert "cluster_label" in scored_df.columns
    assert "risk_flag" in scored_df.columns
    assert len(scored_df) == 300


def test_reproducibility():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=len(FEATURE_COLUMNS), random_state=42)

    df = pd.DataFrame(
        X,
        columns=FEATURE_COLUMNS,
    )
    df["channel_id"] = [f"UC{i:010d}" for i in range(100)]
    df["computed_at"] = "2024-06-15"
    df["insufficient_history"] = False

    model1, _, scored1, _, _ = fit_and_persist(df, random_state=42)
    model2, _, scored2, _, _ = fit_and_persist(df, random_state=42)

    assert (scored1["cluster_id"] == scored2["cluster_id"]).all()


def test_label_clusters_assigns_labels():
    df = pd.DataFrame(
        {
            "channel_id": [f"UC{i:010d}" for i in range(50)],
            "cluster_id": [0] * 25 + [1] * 25,
            "upload_freq_30d": [0.1] * 25 + [0.8] * 25,
            "upload_freq_90d": [0.15] * 25 + [0.7] * 25,
            "freq_trend_ratio": [0.67] * 25 + [1.14] * 25,
            "momentum_ratio": [0.3] * 25 + [2.0] * 25,
            "avg_engagement_rate": [0.05] * 25 + [0.1] * 25,
            "days_since_last_upload": [60] * 25 + [5] * 25,
            "upload_regularity": [20.0] * 25 + [5.0] * 25,
        }
    )

    df["risk_score"] = compute_direct_risk_score(df)

    result = label_clusters_by_risk(df)
    assert "cluster_label" in result.columns
    assert result.groupby("cluster_id")["cluster_label"].nunique().eq(1).all()


def test_assign_risk_flags_produces_three_tiers():
    scores = np.array([0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8, 0.9])
    flags = assign_risk_flags(scores)
    assert "Healthy" in flags
    assert "Watch" in flags
    assert "At-Risk" in flags
    assert sum(flags == "Healthy") >= 1
    assert sum(flags == "At-Risk") >= 1
    assert sum(flags == "Watch") >= 1


def test_assign_risk_flags_monotonic():
    np.random.seed(42)
    scores = np.sort(np.random.uniform(0, 1, 100))
    flags = assign_risk_flags(scores)

    risk_order = {"Healthy": 0, "Watch": 1, "At-Risk": 2}
    mapped = np.array([risk_order[f] for f in flags])
    diffs = np.diff(mapped)
    assert (diffs >= 0).all(), "Risk flags must be monotonic with sorted scores"


def test_compute_direct_risk_score_range():
    df = pd.DataFrame(
        {
            "upload_freq_30d": [0.0, 0.5, 1.0],
            "upload_freq_90d": [0.0, 0.3, 0.5],
            "freq_trend_ratio": [0.0, 1.0, 2.0],
            "momentum_ratio": [0.0, 1.0, 5.0],
            "avg_engagement_rate": [0.0, 0.05, 0.10],
            "days_since_last_upload": [0, 30, 90],
            "upload_regularity": [0, 10, 30],
        }
    )
    scores = compute_direct_risk_score(df)
    assert scores.shape == (3,)
    assert float(np.min(scores)) >= 0.0
    assert float(np.max(scores)) <= 1.0
