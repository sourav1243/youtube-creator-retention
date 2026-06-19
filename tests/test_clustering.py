from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_blobs

from src.config import settings
from src.modeling.cluster import fit_and_persist, label_clusters


def test_clustering_pipeline_deterministic():
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        n_features=7,
        random_state=42,
        cluster_std=2.0,
    )

    df = pd.DataFrame(
        X,
        columns=[
            "upload_freq_30d",
            "upload_freq_90d",
            "momentum_ratio",
            "avg_engagement_rate",
            "days_since_last_upload",
            "upload_regularity",
            "duration_trend",
        ],
    )
    df["channel_id"] = [f"UC{i:010d}" for i in range(300)]
    df["computed_at"] = "2024-06-15"
    df["insufficient_history"] = False

    model, scaler, scored_df, result, _ = fit_and_persist(
        df,
        random_state=settings.clustering.random_state,
    )

    assert model.n_clusters >= 2
    assert model.n_clusters <= 10
    assert "cluster_id" in scored_df.columns
    assert "cluster_label" in scored_df.columns
    assert "risk_flag" in scored_df.columns
    assert len(scored_df) == 300


def test_reproducibility():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=7, random_state=42)

    df = pd.DataFrame(
        X,
        columns=[
            "upload_freq_30d",
            "upload_freq_90d",
            "momentum_ratio",
            "avg_engagement_rate",
            "days_since_last_upload",
            "upload_regularity",
            "duration_trend",
        ],
    )
    df["channel_id"] = [f"UC{i:010d}" for i in range(100)]
    df["computed_at"] = "2024-06-15"
    df["insufficient_history"] = False

    model1, _, scored1, _, _ = fit_and_persist(df, random_state=42)
    model2, _, scored2, _, _ = fit_and_persist(df, random_state=42)

    assert (scored1["cluster_id"] == scored2["cluster_id"]).all()


def test_label_clusters_maps_risk_flags():
    df = pd.DataFrame(
        {
            "channel_id": [f"UC{i:010d}" for i in range(50)],
            "cluster_id": [0] * 25 + [1] * 25,
            "upload_freq_30d": [0.1] * 25 + [0.8] * 25,
            "upload_freq_90d": [0.15] * 25 + [0.7] * 25,
            "engagement_quality": [0.05] * 25 + [0.1] * 25,
            "momentum_ratio": [0.3] * 25 + [2.0] * 25,
            "avg_engagement_rate": [0.05] * 25 + [0.1] * 25,
            "days_since_last_upload": [60] * 25 + [5] * 25,
            "upload_regularity": [20.0] * 25 + [5.0] * 25,
            "duration_trend": [8.0] * 25 + [4.0] * 25,
        }
    )

    result = label_clusters(df)
    risk_flags = result.groupby("cluster_id")["risk_flag"].first()

    assert "At-Risk" in risk_flags.values
    assert "Healthy" in risk_flags.values
